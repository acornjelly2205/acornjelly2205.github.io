Project Status

Project Acorn is currently under active development.

Current version: v0.2 development
Primary language: English
Additional language: Korean

The current development focus is building a bilingual technical knowledge base with structured Study and Project content.

Core Features
English-first technical blog with optional Korean translations
Separate Study and Projects knowledge collections
Topic-based content organization
Unified tags across Study and Projects
Tag statistics and tag-based content browsing
Static search with Pagefind
Responsive light and dark themes
Future support for an interactive knowledge graph
Content Structure
src/content/
├── study/
│   ├── en/
│   └── ko/
├── projects/
│   ├── en/
│   └── ko/
└── pages/
    ├── en/
    └── ko/

English content uses the default URL structure.

/study/
/projects/
/tags/

Korean content uses the /ko/ prefix.

/ko/study/
/ko/projects/
/ko/tags/

Translations are optional. Not every entry is required to have both an English and Korean version.

Tag System

Study and Project entries use shared canonical tag IDs.

tags:
  - cuda
  - tensor-core
  - spmm

Tag labels may be localized, while the canonical tag ID remains identical across languages.

The tag explorer will provide:

Total entries per tag
Study entry count
Project entry count
Tag-specific entry lists
Language-specific tag pages
Clickable tags on content pages
Current Roadmap
v0.2

Add English and Korean locale support

Add language-aware content routing

Add optional translation relationships

Create a shared tag registry

Add Study and Project tag statistics

Add tag-specific content pages

Add clickable tags to content pages

Centralize category and project metadata

Future

Related-content recommendations

Tag relationship visualization

Interactive knowledge graph

Learning progress timeline