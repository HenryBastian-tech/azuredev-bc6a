# VS Code for the Web - Microsoft Foundry

We've generated a simple development environment for you to get started with sample code to create and run an agent.

The Foundry extension provides tools to help you build, test, and deploy AI models and AI Applications directly from VS Code. It offers simplified operations for interacting with your models, agents, and threads without leaving your development environment. Click on the Microsoft Foundry Icon on the left to see more.

Follow the instructions below to get started!

## Open the terminal

Press ``Ctrl-` `` &nbsp; to open a terminal window.

## Prerequisites

- Assign the `Azure AI User` RBAC role to each team member who needs to create or edit agents.
  - This role must be assigned at the project scope
  - Minimum required permissions: `agents/*/read`, `agents/*/action`, `agents/*/delete`
- Deploy a model if you don't have a model deployed in your project. In the left hand activity bar:
  - Open the Microsoft Foundry tab in the navigation bar
  - Under "Models", click "+" to deploy a new AI model, select a model and deploy to Microsoft Foundry
  - In the sample, update `model_deployment_name` name to the deployed model, e.g. `gpt-4o`
  - If you already have models deployed in your project, update `model_deployment_name` name to the model you would like to use

## Create and run your agent locally

Update `agent_name` and `model_deployment_name` in the sample.

To create and run your agent, run the following command:

```bash
python create_agent.py
```

## Continuing on your local desktop

You can keep working locally on VS Code Desktop by clicking "Continue On Desktop..." at the bottom left of this screen. Be sure to take the .env file with you using these steps:

- Right-click the .env file
- Select "Download"
- Move the file from your Downloads folder to the local git repo directory
- For Windows, you will need to rename the file back to .env using right-click "Rename..."

## More examples

Check out [Azure AI Projects client library for Python](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-projects/README.md) for more information on using this SDK.
