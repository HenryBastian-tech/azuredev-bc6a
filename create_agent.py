# Before running:
#   pip install --pre "azure-ai-projects>=2.0.0b1"
#   pip install azure-identity requests
#
# Environment variables you should set:
#   FOUNDRY_PROJECT_ENDPOINT="https://<your-foundry-resource>.services.ai.azure.com/api/projects/<your-project>"
#   FOUNDRY_MODEL_DEPLOYMENT="<your-model-deployment-name>"
#   FOUNDRY_AGENT_NAME="LeanIX-Agent"                       (optional)
#
#   LX_HOST="eu-5.leanix.net"                               (or yourworkspace.leanix.net)
#   LX_API_TOKEN="<LEANIX_TECHNICAL_USER_API_TOKEN>"        (required, NOT base64)

import os
import time
import json
import requests
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

# ----------------------------
# ENV
# ----------------------------
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
FOUNDRY_MODEL = os.getenv("FOUNDRY_MODEL_DEPLOYMENT")
AGENT_NAME = os.getenv("FOUNDRY_AGENT_NAME", "LeanIX-Agent")

LX_HOST = os.getenv("LX_HOST", "eu-5.leanix.net")
LX_API_TOKEN = os.getenv("LX_API_TOKEN")

if not FOUNDRY_ENDPOINT:
    raise RuntimeError("Please set FOUNDRY_PROJECT_ENDPOINT")
if not FOUNDRY_MODEL:
    raise RuntimeError("Please set FOUNDRY_MODEL_DEPLOYMENT")
if not LX_API_TOKEN:
    raise RuntimeError("Please set LX_API_TOKEN")

# ----------------------------
# LeanIX: OAuth2 Client Credentials + Token Cache
# ----------------------------
class LeanIXClient:
    """
    Implements the same flow as the working bash script:
      POST https://{LX_HOST}/services/mtm/v1/oauth2/token
        Authorization: Basic base64("apitoken:<LX_API_TOKEN>")
        grant_type=client_credentials
      -> access_token
      Then:
        Authorization: Bearer <access_token>
        GET https://{LX_HOST}/services/pathfinder/v1/factSheets?...
    """

    def __init__(self, host: str, api_token: str):
        self.host = host
        self.api_token = api_token
        self._token = None
        self._token_exp = 0  # epoch seconds

    def _get_access_token(self) -> str:
        # Reuse token if still valid (30s safety margin)
        now = int(time.time())
        if self._token and now < (self._token_exp - 30):
            return self._token

        url = f"https://{self.host}/services/mtm/v1/oauth2/token"

        # Produces "Basic <base64(apitoken:<token>)>"
        basic = requests.auth._basic_auth_str("apitoken", self.api_token)

        headers = {
            "Authorization": basic,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}

        r = requests.post(url, headers=headers, data=data, timeout=30)
        r.raise_for_status()
        payload = r.json()

        token = payload.get("access_token")
        if not token:
            raise RuntimeError(f"LeanIX token response missing access_token: {payload}")

        expires_in = int(payload.get("expires_in", 300))
        # Keep a safety margin
        self._token = token
        self._token_exp = now + max(60, expires_in - 20)
        return token

    def get_fact_sheets(self, limit: int = 5, fields: str = "id,displayName,type") -> dict:
        token = self._get_access_token()
        url = (
            f"https://{self.host}/services/pathfinder/v1/factSheets"
            f"?limit={limit}&fields={requests.utils.quote(fields)}"
        )
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_fact_sheet_by_id(self, fs_id: str, fields: str = "id,displayName,type") -> dict:
        token = self._get_access_token()
        url = (
            f"https://{self.host}/services/pathfinder/v1/factSheets/{requests.utils.quote(fs_id)}"
            f"?fields={requests.utils.quote(fields)}"
        )
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        r.raise_for_status()
        return r.json()

    def search_fact_sheets(self, query: str, limit: int = 20) -> dict:
        """
        Search strategy:
          1) Try GraphQL (if supported by your tenant/schema)
          2) Fallback to REST list + client-side filtering
        """
        try:
            return self._search_fact_sheets_graphql(query=query, limit=limit)
        except Exception as e:
            raw = self.get_fact_sheets(limit=min(50, max(limit, 10)), fields="id,displayName,type")

            # Be tolerant to different response shapes
            items = raw.get("data", raw)

            results = []
            if isinstance(items, list):
                for el in items:
                    if not isinstance(el, dict):
                        continue
                    name = el.get("displayName")
                    if name and query.lower() in name.lower():
                        results.append({
                            "id": el.get("id"),
                            "displayName": name,
                            "type": el.get("type"),
                        })
                        if len(results) >= limit:
                            break

            return {
                "mode": "fallback_rest_client_filter",
                "query": query,
                "note": f"GraphQL search failed, fallback used: {type(e).__name__}",
                "results": results,
            }

    def _search_fact_sheets_graphql(self, query: str, limit: int) -> dict:
        token = self._get_access_token()
        url = f"https://{self.host}/services/pathfinder/v1/graphql"

        # NOTE: LeanIX GraphQL schemas may differ across tenants/versions.
        # If this doesn't work for your tenant, the code will fall back to REST.
        gql = """
        query ($filter: String!, $first: Int!) {
          allFactSheets(filter: $filter, first: $first) {
            edges {
              node { id displayName type }
            }
          }
        }
        """
        payload = {"query": gql, "variables": {"filter": query, "first": limit}}

        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


# ----------------------------
# Foundry: Project + Agent
# ----------------------------
project_client = AIProjectClient(endpoint=FOUNDRY_ENDPOINT, credential=DefaultAzureCredential())

agent = project_client.agents.create_version(
    agent_name=AGENT_NAME,
    definition=PromptAgentDefinition(
        model=FOUNDRY_MODEL,
        instructions=(
            "You are an enterprise assistant for LeanIX.\n"
            "When you need data from LeanIX, call the provided tools.\n"
            "Prefer 'leanix_search_fact_sheets' for search, 'leanix_get_fact_sheet_by_id' for details, "
            "and 'leanix_get_fact_sheets' for listing.\n"
            "Always produce a concise, structured final answer (bullets or JSON) and include an audit trail "
            "of which tools you called with which parameters."
        ),
    ),
)

openai_client = project_client.get_openai_client()
leanix = LeanIXClient(LX_HOST, LX_API_TOKEN)

# ----------------------------
# Tool definitions (Function Calling)
# ----------------------------
tools = [
    {
        "type": "function",
        "name": "leanix_get_fact_sheets",
        "description": "List FactSheets via LeanIX Pathfinder REST.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 5},
                "fields": {"type": "string", "default": "id,displayName,type"},
            },
        },
    },
    {
        "type": "function",
        "name": "leanix_get_fact_sheet_by_id",
        "description": "Fetch a single FactSheet by ID via Pathfinder REST.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "fields": {"type": "string", "default": "id,displayName,type"},
            },
            "required": ["id"],
        },
    },
    {
        "type": "function",
        "name": "leanix_search_fact_sheets",
        "description": "Search FactSheets (GraphQL when available; otherwise REST fallback).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
            },
            "required": ["query"],
        },
    },
]

def execute_tool(name: str, arguments: dict) -> dict:
    if name == "leanix_get_fact_sheets":
        return leanix.get_fact_sheets(
            limit=int(arguments.get("limit", 5)),
            fields=str(arguments.get("fields", "id,displayName,type")),
        )
    if name == "leanix_get_fact_sheet_by_id":
        return leanix.get_fact_sheet_by_id(
            fs_id=str(arguments["id"]),
            fields=str(arguments.get("fields", "id,displayName,type")),
        )
    if name == "leanix_search_fact_sheets":
        return leanix.search_fact_sheets(
            query=str(arguments["query"]),
            limit=int(arguments.get("limit", 20)),
        )
    return {"error": f"Unknown tool: {name}"}

# ----------------------------
# Responses + tool loop
# ----------------------------
messages = [
    {
        "role": "user",
        "content": (
            "Search for FactSheets with 'SAP' in the displayName, show the top 10, "
            "then fetch details for the first 2 IDs."
        ),
    }
]

for _ in range(8):
    resp = openai_client.responses.create(
        input=messages,
        tools=tools,
        extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
    )

    # If the model produced final text, print it
    if getattr(resp, "output_text", None):
        print(resp.output_text)

    # Extract tool calls (SDK output shape can vary; handle the common case)
    tool_calls = []
    for out in getattr(resp, "output", []) or []:
        if getattr(out, "type", None) == "function_call":
            tool_calls.append(out)

    if not tool_calls:
        break

    # Execute tool calls and append tool results
    for call in tool_calls:
        fn_name = call.name
        fn_args = json.loads(call.arguments or "{}")
        result = execute_tool(fn_name, fn_args)

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": fn_name,
            "content": json.dumps(result),
        })
