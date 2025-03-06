#!/bin/bash

# Create main documentation directories
mkdir -p docs/{api,architecture,development,deployment,examples,guides}

# Create API documentation subdirectories
mkdir -p docs/api/{core,integration,security,examples}

# Create Architecture documentation subdirectories
mkdir -p docs/architecture/{overview,components,integration,flows,design}

# Create Development documentation subdirectories
mkdir -p docs/development/{setup,guidelines,tools,release}

# Create Deployment documentation subdirectories
mkdir -p docs/deployment/{environments,methods,infrastructure,security}

# Create Examples documentation subdirectories
mkdir -p docs/examples/{basic,advanced,integration}

# Create Guides documentation subdirectories
mkdir -p docs/guides/{getting_started,configuration,troubleshooting}

# Move existing files to their appropriate locations
mv docs/TECHNICAL_DOCUMENTATION.md docs/architecture/overview/
mv docs/workflow.md docs/architecture/components/

# Create placeholder files for missing documentation
touch docs/api/core/{graph,memory,context,response}.md
touch docs/api/integration/{langchain,vector_store,web_search}.md
touch docs/api/security/{auth,rate_limiting,security}.md
touch docs/api/examples/{basic,advanced,integration}.md

touch docs/architecture/overview/{overview,principles,components}.md
touch docs/architecture/components/{workflow,memory,context,response}.md
touch docs/architecture/integration/{langchain,vector_store,web_search}.md
touch docs/architecture/flows/{request,memory,context,response}.md
touch docs/architecture/design/{scalability,security,performance,monitoring}.md

touch docs/development/setup/{setup,style,git,structure}.md
touch docs/development/guidelines/{architecture,testing,documentation,security}.md
touch docs/development/tools/{tools,debugging,profiling,review}.md
touch docs/development/release/{release,versioning,deployment,ci_cd}.md

touch docs/deployment/environments/{development,staging,production,config}.md
touch docs/deployment/methods/{docker,kubernetes,manual,ci_cd}.md
touch docs/deployment/infrastructure/{servers,database,caching,monitoring}.md
touch docs/deployment/security/{config,ssl,firewall,access}.md

touch docs/examples/basic/{setup,first_call,configuration}.md
touch docs/examples/advanced/{patterns,optimization,scaling}.md
touch docs/examples/integration/{langchain,vector_store,web_search}.md

touch docs/guides/getting_started/{quickstart,installation,configuration}.md
touch docs/guides/configuration/{environment,security,performance}.md
touch docs/guides/troubleshooting/{common_issues,debugging,recovery}.md

# Set proper permissions
chmod -R 755 docs

echo "Documentation directory structure has been organized successfully!" 