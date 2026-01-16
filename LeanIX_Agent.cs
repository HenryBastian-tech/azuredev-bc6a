// dotnet add package Azure.AI.Projects --version 1.2.*-*
// dotnet add package Azure.Identity
// dotnet add package OpenAI

using System.Collections.Concurrent;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Azure.Identity;
using OpenAI.Responses;

#pragma warning disable OPENAI001

// ========== CONFIG via ENV ==========
// Foundry:
var projectEndpoint = Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")
    ?? "https://<your-foundry-resource>.services.ai.azure.com/api/projects/<your-project>";
var modelDeploymentName = Environment.GetEnvironmentVariable("FOUNDRY_MODEL_DEPLOYMENT")
    ?? "<your-model-deployment-name>";
var agentName = Environment.GetEnvironmentVariable("FOUNDRY_AGENT_NAME") ?? "LeanIX-Agent";

// LeanIX:
var lxHost = Environment.GetEnvironmentVariable("LX_HOST") ?? "eu-5.leanix.net";
var lxApiToken = Environment.GetEnvironmentVariable("LX_API_TOKEN") ?? throw new Exception("Please set LX_API_TOKEN");

// ========== Foundry client ==========
AIProjectClient projectClient = new(endpoint: new Uri(projectEndpoint), tokenProvider: new DefaultAzureCredential());

// Create/Update Agent (version bump when definition changes)
PromptAgentDefinition agentDefinition = new(model: modelDeploymentName)
{
    Instructions =
        "Du bist ein Enterprise Agent für LeanIX.\n" +
        "Wenn du Daten aus LeanIX brauchst, nutze die Tools.\n" +
        "Nutze 'leanix_search_fact_sheets' für Suche, 'leanix_get_fact_sheet_by_id' für Details, " +
        "und 'leanix_get_fact_sheets' für Listen.\n" +
        "Gib am Ende eine knappe, strukturierte Antwort und nenne die verwendeten Tool-Aufrufe."
};

AgentVersion agentVersion = projectClient.Agents.CreateAgentVersion(agentName: agentName, options: new(agentDefinition));
Console.WriteLine($"Agent ready: {agentVersion.Name} v{agentVersion.Version}");

OpenAIResponseClient responseClient = projectClient.OpenAI.GetProjectResponsesClientForAgent(agentVersion);

// ========== Tool definitions (Function Calling) ==========
var tools = new List<ToolDefinition>
{
    ToolDefinition.CreateFunction(
        name: "leanix_get_fact_sheets",
        description: "Listet FactSheets via LeanIX Pathfinder REST.",
        parametersJsonSchema: """
        {
          "type":"object",
          "properties":{
            "limit":{"type":"integer","minimum":1,"maximum":200,"default":5},
            "fields":{"type":"string","default":"id,displayName,type"}
          }
        }
        """
    ),
    ToolDefinition.CreateFunction(
        name: "leanix_get_fact_sheet_by_id",
        description: "Liest ein einzelnes FactSheet per ID via Pathfinder REST.",
        parametersJsonSchema: """
        {
          "type":"object",
          "properties":{
            "id":{"type":"string"},
            "fields":{"type":"string","default":"id,displayName,type"}
          },
          "required":["id"]
        }
        """
    ),
    ToolDefinition.CreateFunction(
        name: "leanix_search_fact_sheets",
        description: "Sucht FactSheets (GraphQL wenn verfügbar; sonst REST-Fallback).",
        parametersJsonSchema: """
        {
          "type":"object",
          "properties":{
            "query":{"type":"string"},
            "limit":{"type":"integer","minimum":1,"maximum":200,"default":20}
          },
          "required":["query"]
        }
        """
    )
};

// ========== Ask the agent ==========
var input = new List<ResponseInputItem>
{
    ResponseInputItem.CreateUserMessage(
        "Suche FactSheets, die 'SAP' im Namen haben, zeige die Top 10 und hole danach Details für die ersten 2 IDs."
    )
};

// ========== Tool loop ==========
var leanIx = new LeanIxClient(lxHost, lxApiToken);
const int maxTurns = 8;

for (int turn = 0; turn < maxTurns; turn++)
{
    OpenAIResponse resp = responseClient.CreateResponse(
        input,
        new ResponseCreationOptions { Tools = tools }
    );

    var toolCalls = resp.Output.OfType<ResponseFunctionToolCall>().ToList();

    // If no tool calls -> final answer
    if (toolCalls.Count == 0)
    {
        Console.WriteLine(resp.GetOutputText());
        break;
    }

    foreach (var call in toolCalls)
    {
        string toolResultJson = await ExecuteToolAsync(call.Name, call.Arguments, leanIx);

        input.Add(ResponseInputItem.CreateToolResult(
            toolCallId: call.Id,
            contentJson: toolResultJson
        ));
    }
}

// =======================
// Tool Execution
// =======================
static async Task<string> ExecuteToolAsync(string toolName, string argsJson, LeanIxClient leanIx)
{
    using var doc = JsonDocument.Parse(argsJson);
    var root = doc.RootElement;

    try
    {
        return toolName switch
        {
            "leanix_get_fact_sheets" => await leanIx.GetFactSheetsAsync(
                limit: root.TryGetProperty("limit", out var l) ? l.GetInt32() : 5,
                fields: root.TryGetProperty("fields", out var f) ? f.GetString()! : "id,displayName,type"
            ),

            "leanix_get_fact_sheet_by_id" => await leanIx.GetFactSheetByIdAsync(
                id: root.GetProperty("id").GetString()!,
                fields: root.TryGetProperty("fields", out var f2) ? f2.GetString()! : "id,displayName,type"
            ),

            "leanix_search_fact_sheets" => await leanIx.SearchFactSheetsAsync(
                query: root.GetProperty("query").GetString()!,
                limit: root.TryGetProperty("limit", out var l2) ? l2.GetInt32() : 20
            ),

            _ => JsonSerializer.Serialize(new { error = $"Unknown tool: {toolName}" })
        };
    }
    catch (Exception ex)
    {
        // Return structured error so the agent can react
        return JsonSerializer.Serialize(new { error = ex.Message, tool = toolName, args = JsonDocument.Parse(argsJson).RootElement });
    }
}

// =======================
// LeanIX Client (Token cache + REST + GraphQL Search)
// =======================
public sealed class LeanIxClient
{
    private readonly string _host;
    private readonly string _apiToken;

    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(30) };

    // Simple token cache per host+token (in-proc)
    private static readonly ConcurrentDictionary<string, TokenCacheEntry> TokenCache = new();

    public LeanIxClient(string host, string apiToken)
    {
        _host = host;
        _apiToken = apiToken;
    }

    public async Task<string> GetFactSheetsAsync(int limit, string fields)
    {
        var accessToken = await GetAccessTokenCachedAsync();
        var url = $"https://{_host}/services/pathfinder/v1/factSheets?limit={limit}&fields={Uri.EscapeDataString(fields)}";
        return await GetAsync(url, accessToken);
    }

    public async Task<string> GetFactSheetByIdAsync(string id, string fields)
    {
        var accessToken = await GetAccessTokenCachedAsync();
        var url = $"https://{_host}/services/pathfinder/v1/factSheets/{Uri.EscapeDataString(id)}?fields={Uri.EscapeDataString(fields)}";
        return await GetAsync(url, accessToken);
    }

    /// <summary>
    /// Search strategy:
    /// 1) Try GraphQL (common in LeanIX Pathfinder).
    /// 2) If GraphQL fails (endpoint/permissions/schema), fallback to REST + client-side filter (small result sizes).
    /// </summary>
    public async Task<string> SearchFactSheetsAsync(string query, int limit)
    {
        // 1) GraphQL attempt
        try
        {
            var gql = await SearchFactSheetsGraphQlAsync(query, limit);
            return gql;
        }
        catch
        {
            // 2) Fallback: REST + client filter
            var raw = await GetFactSheetsAsync(Math.Min(50, Math.Max(limit, 10)), "id,displayName,type");
            using var doc = JsonDocument.Parse(raw);

            IEnumerable<JsonElement> arr = doc.RootElement.ValueKind switch
            {
                JsonValueKind.Array => doc.RootElement.EnumerateArray(),
                JsonValueKind.Object when doc.RootElement.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array
                    => data.EnumerateArray(),
                _ => Array.Empty<JsonElement>()
            };

            var q = query.Trim();
            var results = arr
                .Select(el => new
                {
                    id = el.TryGetProperty("id", out var idEl) ? idEl.GetString() : null,
                    displayName = el.TryGetProperty("displayName", out var dnEl) ? dnEl.GetString() : null,
                    type = el.TryGetProperty("type", out var tEl) ? tEl.GetString() : null
                })
                .Where(x => !string.IsNullOrWhiteSpace(x.displayName) &&
                            x.displayName!.Contains(q, StringComparison.OrdinalIgnoreCase))
                .Take(limit)
                .ToList();

            return JsonSerializer.Serialize(new
            {
                mode = "fallback_rest_client_filter",
                query = q,
                results
            });
        }
    }

    private async Task<string> SearchFactSheetsGraphQlAsync(string query, int limit)
    {
        var accessToken = await GetAccessTokenCachedAsync();
        var url = $"https://{_host}/services/pathfinder/v1/graphql";

        // IMPORTANT:
        // LeanIX GraphQL schema can differ by tenant/version. This query works in many setups,
        // but if your schema differs, adjust "allFactSheets" / fields.
        var gqlQuery = """
        query ($filter: String!, $first: Int!) {
          allFactSheets(filter: $filter, first: $first) {
            edges {
              node {
                id
                displayName
                type
              }
            }
          }
        }
        """;

        var payload = new
        {
            query = gqlQuery,
            variables = new
            {
                filter = query,
                first = limit
            }
        };

        using var req = new HttpRequestMessage(HttpMethod.Post, url);
        req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);
        req.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

        using var resp = await Http.SendAsync(req);
        var body = await resp.Content.ReadAsStringAsync();

        if (!resp.IsSuccessStatusCode)
            throw new HttpRequestException($"LeanIX GraphQL failed: {(int)resp.StatusCode} {resp.ReasonPhrase}\n{body}");

        // Return raw GraphQL JSON so the agent can interpret.
        // Optional: you can normalize into {results:[...]} if preferred.
        return body;
    }

    // -----------------------
    // Token Cache (OAuth2 CC)
    // -----------------------
    private async Task<string> GetAccessTokenCachedAsync()
    {
        var cacheKey = $"{_host}::{_apiToken.GetHashCode()}";
        var now = DateTimeOffset.UtcNow;

        if (TokenCache.TryGetValue(cacheKey, out var entry) && entry.ExpiresAt > now.AddSeconds(30))
            return entry.AccessToken;

        // lock per key to avoid stampede
        var newEntry = await FetchTokenAsync();
        TokenCache[cacheKey] = newEntry;
        return newEntry.AccessToken;
    }

    private async Task<TokenCacheEntry> FetchTokenAsync()
    {
        var tokenUrl = $"https://{_host}/services/mtm/v1/oauth2/token";

        // Basic base64("apitoken:<token>")
        var basicRaw = $"apitoken:{_apiToken}";
        var basicB64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(basicRaw));

        using var req = new HttpRequestMessage(HttpMethod.Post, tokenUrl);
        req.Headers.Authorization = new AuthenticationHeaderValue("Basic", basicB64);
        req.Content = new FormUrlEncodedContent(new Dictionary<string, string>
        {
            ["grant_type"] = "client_credentials"
        });

        using var resp = await Http.SendAsync(req);
        var body = await resp.Content.ReadAsStringAsync();

        if (!resp.IsSuccessStatusCode)
            throw new HttpRequestException($"LeanIX token failed: {(int)resp.StatusCode} {resp.ReasonPhrase}\n{body}");

        using var json = JsonDocument.Parse(body);

        var token = json.RootElement.GetProperty("access_token").GetString();
        var expiresIn = json.RootElement.TryGetProperty("expires_in", out var expEl) ? expEl.GetInt32() : 300;

        if (string.IsNullOrWhiteSpace(token))
            throw new InvalidOperationException("LeanIX access_token missing/empty.");

        // Subtract a small safety margin
        var expiresAt = DateTimeOffset.UtcNow.AddSeconds(Math.Max(60, expiresIn - 20));

        return new TokenCacheEntry(token!, expiresAt);
    }

    // -------------
    // HTTP helpers
    // -------------
    private static async Task<string> GetAsync(string url, string accessToken)
    {
        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);

        using var resp = await Http.SendAsync(req);
        var body = await resp.Content.ReadAsStringAsync();

        if (!resp.IsSuccessStatusCode)
            throw new HttpRequestException($"LeanIX call failed: {(int)resp.StatusCode} {resp.ReasonPhrase}\n{body}");

        return body;
    }

    private readonly record struct TokenCacheEntry(string AccessToken, DateTimeOffset ExpiresAt);
}
