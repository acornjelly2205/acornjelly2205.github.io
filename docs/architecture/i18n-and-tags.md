Internationalization and Tag Architecture
Decision Summary

Project Acorn uses English as its default language and Korean as an optional secondary language.

Study and Project entries share a canonical tag system. Tags are used both for content navigation and as the foundation of a future knowledge graph.

Route Strategy

English routes do not use a locale prefix.

/study/
/projects/
/tags/

Korean routes use /ko/.

/ko/study/
/ko/projects/
/ko/tags/
Translation Strategy

Translations are optional.

A translated pair shares a stable translationKey.

lang: en
translationKey: tc-bell-overview
lang: ko
translationKey: tc-bell-overview

The system must not assume that every entry has a translation.

Tag Strategy

Frontmatter uses canonical tag IDs.

tags:
  - cuda
  - tensor-core
  - spmm

Canonical IDs must:

Use lowercase letters
Use hyphens between words
Remain identical across languages
Avoid localized text
Remain stable after publication

Localized display labels are stored in a shared registry.

Tag Statistics

For each canonical tag, the tag index calculates:

Total number of entries
Number of Study entries
Number of Project entries

Only non-draft entries in the selected language are counted.

A single entry contributes at most one count per tag, even if a tag is accidentally repeated in its frontmatter.

Tag Pages

The English tag index is available at:

/tags/

The Korean tag index is available at:

/ko/tags/

Tag detail pages use the canonical tag ID.

/tags/cuda/
/ko/tags/cuda/

Each tag detail page displays:

Localized tag label
Total entry count
Study count
Project count
Study entries
Project entries
Pagination when necessary
Future Knowledge Graph

Canonical tags will later become nodes in the Project Acorn knowledge graph.

Potential relationships include:

Entry → Tag
Project → Study note
English entry → Korean translation
Tag → Related tag
Project → Research topic