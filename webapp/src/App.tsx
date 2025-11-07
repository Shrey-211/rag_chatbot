import { FormEvent, useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

type StatsResponse = {
  total_documents: number;
  vectorstore_provider: string;
  llm_provider: string;
  embedding_provider: string;
  embedding_dimension: number;
};

type Metadata = {
  [key: string]: unknown;
  source?: string;
  document_id?: string;
  chunk_index?: number;
  total_chunks?: number;
};

type SourceDocument = {
  document_id: string;
  filename?: string;
  content_type?: string;
  score: number;
  num_chunks: number;
  chunks?: Array<{ id: string; content: string; score: number }>;
  has_file: boolean;
};

type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
  sources?: SourceDocument[];
};

type IndexedDocument = {
  documentId: string;
  fileName?: string;
  message: string;
  numChunks: number;
  uploadedAt: string;
  metadata?: Metadata;
  hasFile?: boolean;
};

type QueryApiResponse = {
  answer: string;
  sources?: SourceDocument[];
};

type IndexApiResponse = {
  success: boolean;
  document_id: string;
  num_chunks: number;
  message: string;
};

const formatError = (error: unknown) => {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return "Unexpected error. Check the browser console for details.";
};

const parseMetadata = (raw: string): Metadata | undefined => {
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }

  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Metadata;
    }
    throw new Error("Metadata must be a JSON object");
  } catch (error) {
    throw new Error(`Invalid metadata JSON: ${formatError(error)}`);
  }
};

const getDocumentLabel = (doc: IndexedDocument): string => {
  if (doc.fileName) {
    return doc.fileName;
  }
  const candidate = doc.metadata?.source;
  if (typeof candidate === "string" && candidate.trim().length > 0) {
    return candidate;
  }
  return "Text snippet";
};

// Removed getSourceLabel - now using source.filename or document_id directly

const App = () => {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [textToIndex, setTextToIndex] = useState("");
  const [textMetadata, setTextMetadata] = useState("{}");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const chatMessagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);
  
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<"chat" | "documents">("chat");
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);

  const refreshStats = async () => {
    setStatsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      if (!response.ok) {
        throw new Error(`Stats request failed (${response.status})`);
      }
      const data: StatsResponse = await response.json();
      setStats(data);
    } catch (error) {
      console.error("Stats error", error);
    } finally {
      setStatsLoading(false);
    }
  };

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents?limit=100`);
      if (!response.ok) {
        throw new Error(`Documents request failed (${response.status})`);
      }
      const data = await response.json();
      const docs: IndexedDocument[] = data.documents.map((doc: any) => ({
        documentId: doc.document_id,
        fileName: doc.filename,
        message: `${doc.num_chunks} chunks`,
        numChunks: doc.num_chunks,
        uploadedAt: doc.indexed_at,
        metadata: doc.metadata
      }));
      setDocuments(docs);
    } catch (error) {
      console.error("Failed to fetch documents", error);
    }
  };

  useEffect(() => {
    refreshStats().catch((error) => {
      console.error("Failed to load stats", error);
    });
    fetchDocuments().catch((error) => {
      console.error("Failed to load documents", error);
    });
  }, []);

  useEffect(() => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  useEffect(() => {
    const interval = setInterval(() => {
      refreshStats().catch((error) => {
        console.error("Failed to auto-refresh stats", error);
      });
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleIndexText = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!textToIndex.trim()) {
      setUploadError("Please provide text to index.");
      setUploadSuccess(null);
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const metadata = parseMetadata(textMetadata);
      const response = await fetch(`${API_BASE_URL}/index`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: textToIndex,
          metadata
        })
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Indexing failed (${response.status})`);
      }

      const data: IndexApiResponse = await response.json();
      setUploadSuccess(data.message ?? "Indexed text successfully.");
      setTextToIndex("");
      refreshStats().catch((error) => console.error("Failed to refresh stats", error));
      fetchDocuments().catch((error) => console.error("Failed to fetch documents", error));
    } catch (error) {
      console.error("Index text error", error);
      setUploadError(formatError(error));
    } finally {
      setUploading(false);
    }
  };

  const handleFileUpload = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const fileInput = fileInputRef.current;
    const file = fileInput?.files?.[0];

    if (!file) {
      setUploadError("Select a file to upload.");
      setUploadSuccess(null);
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/index/file`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `File upload failed (${response.status})`);
      }

      const data: IndexApiResponse = await response.json();
      setUploadSuccess(data.message ?? `Uploaded ${file.name}`);
      if (fileInput) {
        fileInput.value = "";
      }
      setSelectedFileName(null);
      refreshStats().catch((error) => console.error("Failed to refresh stats", error));
      fetchDocuments().catch((error) => console.error("Failed to fetch documents", error));
    } catch (error) {
      console.error("File upload error", error);
      setUploadError(formatError(error));
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (documentId: string, documentName: string) => {
    if (!confirm(`Are you sure you want to delete "${documentName}"?\n\nThis will permanently remove:\n• The document from the database\n• The uploaded file (if exists)\n• All embeddings and chunks\n\nThis action cannot be undone.`)) {
      return;
    }

    setDeletingDocId(documentId);
    
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Delete failed (${response.status})`);
      }

      const data = await response.json();
      
      // Remove document from local state
      setDocuments((prev) => prev.filter((doc) => doc.documentId !== documentId));
      
      // Refresh stats
      refreshStats().catch((error) => console.error("Failed to refresh stats", error));
      
      // Show success message
      setUploadSuccess(data.message || "Document deleted successfully");
      setUploadError(null);
      
      // Clear success message after 3 seconds
      setTimeout(() => setUploadSuccess(null), 3000);
    } catch (error) {
      console.error("Delete error", error);
      setUploadError(formatError(error));
      setUploadSuccess(null);
    } finally {
      setDeletingDocId(null);
    }
  };

  const handleQuerySubmit = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setChatError("Enter a question to query the knowledge base.");
      return;
    }

    setChatError(null);
    setChatMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setQuery("");
    setChatLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: trimmed,
          top_k: topK,
          include_sources: true
        })
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Query failed (${response.status})`);
      }

      const data: QueryApiResponse = await response.json();
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: data.answer,
        sources: data.sources ?? []
      };

      setChatMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Query error", error);
      setChatError(formatError(error));
      setChatMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: "The assistant could not generate a response. Please try again."
        }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="app">
      <aside className={`sidebar ${sidebarOpen ? "sidebar--open" : ""}`}>
        <div className="sidebar__header">
          <button
            className="sidebar__toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? "←" : "→"}
          </button>
          {sidebarOpen && <h2 className="sidebar__title">RAG Chatbot</h2>}
        </div>

        {sidebarOpen && (
          <div className="sidebar__content">
            <div className="sidebar__tabs">
              <button
                className={`sidebar__tab ${activeTab === "chat" ? "sidebar__tab--active" : ""}`}
                onClick={() => setActiveTab("chat")}
              >
                💬 Chat
              </button>
              <button
                className={`sidebar__tab ${activeTab === "documents" ? "sidebar__tab--active" : ""}`}
                onClick={() => setActiveTab("documents")}
              >
                📄 Documents
              </button>
            </div>

            {activeTab === "documents" && (
              <div className="sidebar__section">
                <div className="sidebar__stats">
                  {stats && (
                    <>
                      <div className="stat-item">
                        <span className="stat-label">Documents</span>
                        <span className="stat-value">{stats.total_documents}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">Provider</span>
                        <span className="stat-value">{stats.llm_provider}</span>
                      </div>
                    </>
                  )}
                </div>

                <div className="upload-section">
                  <h3 className="upload-section__title">Upload Documents</h3>
                  
                  <form className="upload-form" onSubmit={handleFileUpload}>
                    <input
                      ref={fileInputRef}
                      type="file"
                      className="upload-form__file"
                      accept=".pdf,.doc,.docx,.txt,.csv,.md"
                      disabled={uploading}
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        setSelectedFileName(file ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)` : null);
                      }}
                    />
                    {selectedFileName && (
                      <div className="file-info">
                        <span className="file-info__name">📄 {selectedFileName}</span>
                      </div>
                    )}
                    <button type="submit" className="button button--full" disabled={uploading || !selectedFileName}>
                      {uploading ? "Uploading..." : "Upload File"}
                    </button>
                  </form>

                  <div className="divider" />

                  <form className="upload-form" onSubmit={handleIndexText}>
                    <textarea
                      className="upload-form__textarea"
                      rows={4}
                      placeholder="Paste text to index..."
                      value={textToIndex}
                      onChange={(event) => setTextToIndex(event.target.value)}
                      disabled={uploading}
                    />
                    <button type="submit" className="button button--full" disabled={uploading}>
                      {uploading ? "Indexing..." : "Index Text"}
                    </button>
                  </form>

                  {(uploadError || uploadSuccess) && (
                    <div className={`alert ${uploadError ? "alert--error" : "alert--success"}`}>
                      {uploadError ?? uploadSuccess}
                    </div>
                  )}
                </div>

                <div className="documents-list">
                  <div className="documents-list__header">
                    <h3>Indexed Documents</h3>
                    <button
                      className="button button--small button--secondary"
                      onClick={fetchDocuments}
                    >
                      Refresh
                    </button>
                  </div>
                  <div className="documents-list__content">
                    {documents.length === 0 ? (
                      <div className="documents-list__empty">No documents yet</div>
                    ) : (
                      documents.map((doc) => (
                        <div key={doc.documentId} className="document-item">
                          <div className="document-item__header">
                            <div className="document-item__name">{getDocumentLabel(doc)}</div>
                            <div className="document-item__actions">
                              {doc.hasFile && (
                                <a
                                  href={`${API_BASE_URL}/documents/${doc.documentId}/file`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="document-item__icon-button"
                                  onClick={(e) => e.stopPropagation()}
                                  title="View file"
                                >
                                  📥
                                </a>
                              )}
                              <button
                                className="document-item__icon-button document-item__icon-button--delete"
                                onClick={() => handleDeleteDocument(doc.documentId, getDocumentLabel(doc))}
                                disabled={deletingDocId === doc.documentId}
                                title="Delete document"
                              >
                                {deletingDocId === doc.documentId ? "⏳" : "🗑️"}
                              </button>
                            </div>
                          </div>
                          <div className="document-item__meta">
                            {doc.numChunks} chunks • {new Date(doc.uploadedAt).toLocaleDateString()}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === "chat" && (
              <div className="sidebar__section">
                <div className="chat-settings">
                  <label className="chat-settings__label">
                    Top K Results
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={topK}
                      onChange={(e) => {
                        const value = Number(e.target.value);
                        setTopK(Number.isFinite(value) && value > 0 ? value : 5);
                      }}
                      className="chat-settings__input"
                    />
                  </label>
                </div>
              </div>
            )}
          </div>
        )}
      </aside>

      <main className="main-content">
        <div className="chat-container" ref={chatContainerRef}>
          <div className="chat-messages" ref={chatMessagesEndRef}>
            {chatMessages.length === 0 ? (
              <div className="chat-empty">
                <div className="chat-empty__icon">🤖</div>
                <h1 className="chat-empty__title">How can I help you today?</h1>
                <p className="chat-empty__subtitle">Ask questions about your indexed documents</p>
              </div>
            ) : (
              chatMessages.map((message, index) => (
                <div key={index} className={`message message--${message.role}`}>
                  <div className="message__avatar">
                    {message.role === "user" ? "👤" : message.role === "assistant" ? "🤖" : "⚠️"}
                  </div>
                  <div className="message__content">
                    <div className="message__text">{message.content}</div>
                    {message.sources && message.sources.length > 0 && (
                      <details className="message__sources">
                        <summary className="message__sources-summary">
                          📚 {message.sources.length} source{message.sources.length > 1 ? "s" : ""}
                        </summary>
                        <div className="message__sources-list">
                          {message.sources.map((source) => (
                            <div key={source.document_id} className="source-item">
                              <div className="source-item__header">
                                <div className="source-item__title-row">
                                  <span className="source-item__name">
                                    {source.filename || source.document_id.substring(0, 12)}
                                  </span>
                                  {source.has_file && (
                                    <a
                                      href={`${API_BASE_URL}/documents/${source.document_id}/file`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="source-item__download"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      📥 View
                                    </a>
                                  )}
                                </div>
                                <div className="source-item__meta">
                                  <span className="source-item__score">Score: {source.score.toFixed(3)}</span>
                                  <span className="source-item__chunks">{source.num_chunks} chunk{source.num_chunks !== 1 ? "s" : ""}</span>
                                </div>
                              </div>
                              {source.chunks && source.chunks.length > 0 && (
                                <div className="source-item__preview">
                                  <div className="source-item__preview-label">Preview:</div>
                                  {source.chunks.map((chunk, idx) => (
                                    <div key={chunk.id} className="source-item__chunk">
                                      {chunk.content}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              ))
            )}
            {chatLoading && (
              <div className="message message--assistant">
                <div className="message__avatar">🤖</div>
                <div className="message__content">
                  <div className="message__loading">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatMessagesEndRef} />
          </div>

          {chatError && (
            <div className="chat-error">
              <div className="alert alert--error">{chatError}</div>
            </div>
          )}

          <form className="chat-input-form" onSubmit={handleQuerySubmit}>
            <div className="chat-input-wrapper">
              <textarea
                className="chat-input"
                rows={1}
                placeholder="Message RAG Chatbot..."
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  event.target.style.height = "auto";
                  event.target.style.height = `${event.target.scrollHeight}px`;
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    if (query.trim() && !chatLoading) {
                      handleQuerySubmit();
                    }
                  }
                }}
                disabled={chatLoading}
              />
              <button
                type="submit"
                className="chat-send-button"
                disabled={chatLoading || !query.trim()}
                aria-label="Send message"
              >
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <path d="M18 2L9 11M18 2L12 18L9 11M18 2L2 8L9 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>
            <div className="chat-input-footer">
              <span className="chat-input-footer__text">Press Enter to send, Shift+Enter for new line</span>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
};

export default App;
