# API Keys

The different API surfaces that **Routable** exposes are all protected by an API Key:

- the Inference and Metrics API surfaces will respond with an error if the API Key is not provided or incorrect
- the Admin API surface will not respond at all if an API Key is not provided
