# Vectorstore

Ein skalierbarer Vektorspeicher mit FAISS und ChromaDB für schnelle Ähnlichkeitssuche und persistente Speicherung.

## Übersicht

Das Vectorstore-System bietet eine vollwertige Vektordatenbank mit folgenden Kernkomponenten:

1. **FAISS (CPU/GPU)** - Schnelle Ähnlichkeitssuche in hochdimensionalen Vektorräumen
2. **ChromaDB** - Persistente Speicherung von Vektoren, IDs und Metadaten
3. **FastAPI-Server** - REST-API für Web-Anwendungen und Integrationen
4. **MCP-Server** - Model Context Protocol Integration für KI-Modelle
5. **Synchronisationsmanager** - Thread-sichere Operationen und Datenkonsistenz
6. **Concurrency-Manager** - Effiziente gleichzeitige Operationen
7. **Backup-System** - Automatische Sicherung und Wiederherstellung

## Technologie-Stack

- **Python**: 3.10+
- **Vektorspeicher**: FAISS (GPU/CPU) + ChromaDB
- **Web-Framework**: FastAPI
- **Datenvalidierung**: Pydantic
- **Asynchrone Verarbeitung**: asyncio
- **Konfigurationsmanagement**: tomllib
- **Werkzeuge**: uvicorn, tqdm

## Installation

### Voraussetzungen

- Python 3.10+
- uv (Python-Paketmanager)
- Optional: CUDA-kompatible GPU für GPU-Beschleunigung

### Einrichtung

1. Klonen des Repositories:
```bash
git clone <repository-url>
cd Vectorstore
```

2. Erstellen der virtuellen Umgebung:
```bash
uv venv
```

3. Aktivieren der virtuellen Umgebung:
```bash
# Windows
.venv\Scripts\activate

# Unix/MacOS
source .venv/bin/activate
```

4. Installieren der Abhängigkeiten:
```bash
uv sync
```

5. Installieren der GPU-Unterstützung (optional / nicht für Windows verfügbar):
```bash
uv add --extra gpu faiss-gpu
```

## Verwendung

### Kommandozeile

Das System kann direkt über die Kommandozeile verwendet werden:

```python
from src.knowledge_base import KnowledgeBase, KnowledgeBaseEntry
import numpy as np

# KnowledgeBase erstellen
kb = KnowledgeBase(
    persist_directory="./data/knowledge_base",
    embedding_dimension=1024,
    use_gpu=False
)

# Eintrag hinzufügen
entry = KnowledgeBaseEntry(
    id="entry_1",
    text="Beispieltext",
    embedding=np.random.rand(1024),
    metadata={"category": "example"},
    cluster_id=0
)
kb.add_entry(entry)

# Suche durchführen
query_embedding = np.random.rand(1024)
results = kb.search(query_embedding, k=5)
```

### FastAPI Server

Starten Sie den REST-API-Server:

```bash
python -m src.server.fastapi_server
```

Mit GPU-Unterstützung:
```bash
export USE_GPU=true
python -m src.server.fastapi_server
```

Der Server bietet folgende Endpunkte:

- `GET /health` - Gesundheitsprüfung
- `POST /vectors/search` - Vektorsuche
- `POST /vectors/add` - Vektor hinzufügen
- `POST /vectors/batch-add` - Batch-Vektoren hinzufügen
- `GET /vectors/{vector_id}` - Vektor abrufen
- `DELETE /vectors/{vector_id}` - Vektor löschen
- `GET /statistics` - Statistiken abrufen
- `POST /backup` - Backup erstellen
- `POST /clear` - KnowledgeBase leeren

### MCP Server

Starten Sie den MCP-Server für KI-Integration:

```bash
python -m src.server.mcp_server
```

Der MCP-Server bietet folgende Werkzeuge:

- `search_knowledge_base` - Suche im Wissensspeicher
- `add_vector` - Vektor hinzufügen
- `get_vector` - Vektor abrufen
- `delete_vector` - Vektor löschen
- `get_statistics` - Statistiken abrufen
- `clear_knowledge_base` - Wissensspeicher leeren

## Projektstruktur

```
Vectorstore/
├── src/
│   ├── __init__.py
│   ├── knowledge_base.py     # Haupt-KnowledgeBase mit FAISS+ChromaDB
│   └── server/
│       ├── fastapi_server.py # FastAPI REST-Server
│       └── mcp_server.py     # MCP Server für KI-Integration
├── pyproject.toml
├── README.md
├── uv.lock
└── .gitignore
```

## Konfiguration

Die Konfiguration erfolgt über die `pyproject.toml` Datei:

```toml
[tool.knowledgebase]
# Server Configuration
server_host = "0.0.0.0"
server_port = 8000

# Database Configuration
database_persist_directory = "./data/knowledge_base"
database_embedding_dimension = 1024
database_use_gpu = false
database_index_type = "flat"

# Performance Configuration
performance_max_workers = 4
performance_batch_size = 10
performance_cache_ttl = 300

# Search Configuration
search_max_results = 10
search_similarity_threshold = 0.1
search_enable_caching = true

# MCP Configuration
mcp_enabled = true
mcp_host = "localhost"
mcp_port = 8001
mcp_max_results = 10
mcp_enable_caching = true
```

## Kernkomponenten im Detail

### KnowledgeBase

Die Hauptklasse kombiniert FAISS und ChromaDB:

```python
from src.knowledge_base import KnowledgeBase

# Initialisierung
kb = KnowledgeBase(
    persist_directory="./data/knowledge_base",
    embedding_dimension=1024,
    use_gpu=False,
    index_type="flat"
)

# Einträge verwalten
kb.add_entry(entry)
kb.add_entries_batch(entries)
results = kb.search(query_embedding, k=10)
```

**Funktionen:**
- Hybridspeicher: FAISS für schnelle Suche, ChromaDB für Persistenz
- Metadatenverwaltung und Filterung
- Thread-sichere Operationen
- Batch-Verarbeitung
- Backup- und Restore-Funktionen

### FAISSIndex

Verwaltet FAISS-Indizes für schnelle Ähnlichkeitssuche:

```python
from src.knowledge_base import FAISSIndex

# Index erstellen
index = FAISSIndex(
    dimension=1024,
    use_gpu=False,
    index_type="flat"
)

# Einträge hinzufügen
index.add_embeddings(embeddings)

# Suche durchführen
distances, indices = index.search(query_embedding, k=10)
```

**Unterstützte Index-Typen:**
- `flat` - Exakte Suche, hohe Genauigkeit
- `ivf` - Inverted File System, schneller für große Datensätze
- `pq` - Product Quantization, speichereffizient

### GPU-Beschleunigung

Für GPU-Beschleunigung:

1. Installieren Sie die GPU-Version von FAISS:
```bash
uv add --extra gpu faiss-gpu
```

2. Aktivieren Sie GPU in der Konfiguration:
```python
kb = KnowledgeBase(use_gpu=True)
```

3. Starten Sie den Server mit GPU-Unterstützung:
```bash
export USE_GPU=true
python -m src.server.fastapi_server
```

**Voraussetzungen für GPU-Nutzung:**
- CUDA-kompatible GPU
- Installierte CUDA-Toolkit
- Faiss GPU-Paket (`faiss-gpu>=1.7.0`)
- Linux

### ChromaDBManager

Verwaltet die persistente Speicherung:

```python
from src.knowledge_base import ChromaDBManager

# Manager initialisieren
chroma_db = ChromaDBManager("./data/chroma_db")

# Einträge hinzufügen
chroma_db.add_entry(entry)

# Suche mit Filtern
results = chroma_db.search(query_embedding, k=10, filters={"category": "tech"})
```

**Funktionen:**
- Persistente Speicherung
- Metadatenfilterung
- Batch-Operationen
- Sammlungsstatistiken

### Embedding-Generierung

Das System unterstützt jetzt automatische Embedding-Generierung mit zwei Methoden:

#### 1. Ollama mit BGE-M3 (empfohlen)
Verwendet Ollama mit dem BGE-M3 Modell für OpenAI-kompatible Embeddings.

#### 2. Sentence Transformers
Verwendet das `paraphrase-multilingual-MiniLM-L12-v2` Modell für mehrsprachige Embeddings.

#### Konfiguration über .env Datei:
```env
# Embedding Methode auswählen
KB_EMBEDDING_METHOD=ollama  # Options: ollama, sentence-transformer

# Sentence Transformer Modell
KB_SENTENCE_TRANSFORMER_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Ollama Konfiguration
KB_OLLAMA_URL=http://localhost:11434
KB_BGE_MODEL=bge-m3
```

#### Verwendung mit automatischer Embedding-Generierung:
```python
from src.knowledge_base import KnowledgeBase, create_knowledge_base_from_clusters

# KnowledgeBase mit automatischer Embedding-Generierung
kb = KnowledgeBase()

# Texte direkt hinzufügen - Embeddings werden automatisch generiert
texts = ["Dies ist ein Beispieltext", "Noch ein Text"]
for i, text in enumerate(texts):
    entry = KnowledgeBaseEntry(
        id=f"entry_{i}",
        text=text,
        embedding=kb.embedding_manager.generate_embeddings([text])[0],
        metadata={"source": "auto-generated"},
        cluster_id=0
    )
    kb.add_entry(entry)

# Oder mit Clustern arbeiten
cluster_data = [
    (0, ["Text 1 aus Cluster 0", "Text 2 aus Cluster 0"]),
    (1, ["Text 1 aus Cluster 1"])
]

# Embeddings werden automatisch generiert
kb = create_knowledge_base_from_clusters(cluster_data)
```

#### Verwendung mit manuell generierten Embeddings:
```python
from src.embedding_manager import EmbeddingManager

# Embedding Manager erstellen
embedding_manager = EmbeddingManager(method='sentence-transformer')

# Embeddings generieren
texts = ["Dies ist ein Beispieltext", "Noch ein Text"]
embeddings = embedding_manager.generate_embeddings(texts)

# Zur KnowledgeBase hinzufügen
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    entry = KnowledgeBaseEntry(
        id=f"entry_{i}",
        text=text,
        embedding=embedding,
        metadata={"source": "sentence-transformer"},
        cluster_id=0
    )
    kb.add_entry(entry)
```

#### Embedding-Methoden im Detail:

**Ollama mit BGE-M3 (OpenAI-kompatibel):**
- Dimension: 1024
- Vorteile: Schnell, OpenAI-kompatibel, einfache API
- Voraussetzung: Läuft Ollama mit `bge-m3:latest` Modell
- Verwendung: `from openai import OpenAI` mit `base_url='http://localhost:11434/v1/'`

**Sentence Transformers:**
- Dimension: 384 (für paraphrase-multilingual-MiniLM-L12-v2)
- Vorteile: Offline, mehrsprachig
- Voraussetzung: `sentence-transformers` Bibliothek installiert

**Batch-Verarbeitung:**
```python
# Große Textmengen effizient verarbeiten
texts = ["Text 1", "Text 2", "Text 3", ...]

# Alle Embeddings auf einmal generieren
embeddings = kb.embedding_manager.generate_embeddings(texts)
```

**Hinweise:**
- Die Embedding-Methode wird über die Umgebungsvariable `KB_EMBEDDING_METHOD` gesteuert
- Alle Embeddings müssen die gleiche Dimension haben
- Die Dimension wird automatisch aus dem gewählten Modell ermittelt
- Für optimale Ergebnisse sollten die Embeddings mit demselben Modell generiert werden

### SyncManager

Stellt Datenkonsistenz zwischen FAISS und ChromaDB sicher:

```python
# Thread-sichere Operationen
sync_manager.add_vector(vector, text, metadata, cluster_id)
sync_manager.delete_vector(vector_id)
sync_manager.rebuild_faiss_index()
```

## API-Dokumentation

### FastAPI Endpunkte

#### Gesundheitsprüfung
```
GET /health
```

**Antwort:**
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "faiss_index_built": true,
  "total_entries": 42,
  "use_gpu": false,
  "index_type": "flat"
}
```

#### Vektorsuche
```
POST /vectors/search
```

**Anfrage:**
```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10,
  "filters": {"category": "tech"},
  "index_type": "flat"
}
```

**Antwort:**
```json
{
  "results": [
    {
      "id": "vec_123",
      "text": "Beispieltext",
      "similarity_score": 0.95,
      "rank": 1,
      "metadata": {"category": "tech"},
      "cluster_id": 0
    }
  ],
  "total_results": 5,
  "query_dimension": 1024,
  "index_type": "flat",
  "use_gpu": false
}
```

#### Vektor hinzufügen
```
POST /vectors/add
```

**Anfrage:**
```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "text": "Beispieltext",
  "metadata": {"category": "tech"},
  "cluster_id": 0
}
```

**Antwort:**
```json
{
  "id": "vec_123",
  "message": "Vector added successfully",
  "total_entries": 43,
  "index_type": "flat",
  "use_gpu": false
}
```

#### Statistiken
```
GET /statistics
```

**Antwort:**
```json
{
  "total_entries": 42,
  "unique_clusters": 5,
  "cluster_distribution": {0: 10, 1: 8, 2: 12, 3: 7, 4: 5},
  "faiss_index_built": true,
  "use_gpu": false,
  "index_type": "flat",
  "chroma_stats": {
    "total_entries": 42,
    "persist_directory": "./data/knowledge_base/chroma_db"
  }
}
```

## Performance-Optimierung

### GPU-Beschleunigung

Für GPU-Beschleunigung:

1. Installieren Sie die GPU-Version von FAISS:
```bash
uv add faiss-gpu
```

2. Aktivieren Sie GPU in der Konfiguration:
```python
kb = KnowledgeBase(use_gpu=True)
```

### Batch-Verarbeitung

Nutzen Sie Batch-Operationen für bessere Performance:

```python
# Batch-Einträge hinzufügen
kb.add_entries_batch(entries)

# Batch-Suche
results = kb.search_batch([query1, query2, query3], k=10)
```

### Caching

Das System unterstützt Caching für häufig verwendete Operationen:

```python
# Caching in der Konfiguration aktivieren
search_cache_ttl = 300  # 5 Minuten
enable_caching = true
```

## Fehlerbehandlung

Das System umfassende Fehlerbehandlung:

```python
try:
    results = kb.search(query_embedding, k=10)
    if not results:
        print("Keine Ergebnisse gefunden")
except Exception as e:
    print(f"Suchfehler: {e}")
```

## Wartung

### Backup

Erstellen Sie regelmäßige Backups:

```python
# Backup erstellen
kb.save("./backup")

# Backup laden
kb.load("./backup")
```

### Monitoring

Überwachen Sie die Performance:

```python
# Statistiken abrufen
stats = kb.get_statistics()
print(f"KnowledgeBase-Größe: {stats['total_entries']}")
print(f"Eindeutige Cluster: {stats['unique_clusters']}")
```

## Troubleshooting

### Häufige Probleme

1. **FAISS GPU nicht verfügbar**
   - Installieren Sie die CPU-Version: `uv add faiss-cpu`
   - Setzen Sie `use_gpu=False` in der Konfiguration

2. **ChromaDB Fehler**
   - Löschen Sie den persistenz-Ordner und starten Sie neu
   - Überprüfen Sie die Schreibrechte

3. **Speicherprobleme**
   - Reduzieren Sie die Batch-Größe
   - Verwenden Sie Product Quantization (`index_type="pq"`)
   - Aktivieren Sie Caching

### Logs

Aktivieren Sie detaillierte Logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.
