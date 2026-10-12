# what is collection in qdrant

A Collection is the primary storage unit in Qdrant that stores vectors (points). It is conceptually similar to a table in a relational database like PostgreSQL. Each collection contains points, where each point has an ID, one or more vectors, and optional payload (metadata).

Qdrant Server

└── Collection: documents
      ├── Point
      │    ID: 1
      │    Vector: [0.1,0.2,...]
      │    Payload:
      │      title: Introduction to AI
      │      category: AI
      │
      └── Point
           ID: 2
           Vector: [0.5,0.6,...]
           Payload:
             title: Python Basics
             category: Python