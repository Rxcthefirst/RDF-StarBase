repo/
  README.md
  CHANGELOG.md

  docs/
    agent/
      AGENT_PROMPT.md
      CONTEXT_INDEX.md        # "read these first" map for agents
    plans/
      roadmap.md              # north-star milestones
      current_sprint.md       # exactly what the agent should work on now
      backlog.md              # ideas, not active
    architecture/
      overview.md             # diagrams + component responsibilities
      deployment.md           # compose/k8s assumptions, ports, volumes
      security_model.md       # authN/authZ, RBAC roles, audit events
    contracts/
      api/
        openapi.yaml          # or markdown contract + examples
        examples/
      data/
        rdfstar_profile.md    # what RDF*/SPARQL* subset is supported
    adr/
      0001-initial-architecture.md
      0002-auth-approach.md
    runbooks/
      local_dev.md
      build_release.md
      troubleshooting.md

  config/
    default.yaml
    local.yaml.example
    prod.yaml.example

  deploy/
    docker/
      Dockerfile.api
      Dockerfile.ui
    compose/
      docker-compose.yml
    k8s/
      base/
      overlays/
    helm/                      # optional later

  src/
    rdf_starbase/              # Core engine (unchanged package name)
      storage/                 # Storage layer modules
    api/                       # REST API layer (separated from engine)
      __init__.py
      auth.py                  # Authentication/authorization
      web.py                   # FastAPI app
      repository_api.py        # Repository management endpoints
      ai_grounding.py          # AI/RAG endpoints
    ui/                        # Web UI (renamed from frontend)

  tests/
    # Tests organized by feature, not strictly by engine/api
