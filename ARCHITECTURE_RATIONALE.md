# Architecture Rationale

- **VMD only:** one decomposition is sufficient for the lead's shared-decomposition requirement and keeps implementation feasible.
- **All modes by default:** deleting unusual modes can remove dysarthric evidence. Mode deletion is therefore only an ablation.
- **CNN instead of ViT:** TORGO has few independent speakers; a lightweight CNN is simpler and lower-risk.
- **Pure-PyTorch GCN:** a ≤256-node graph can use normalized adjacency directly, avoiding extra graph-framework installation complexity.
- **Token-level attention:** cross-attention over one global vector per branch is not meaningful enough. 16 Mel tokens and 17 nonlinear tokens give real cross-representation interactions.
- **One attention block:** enough to demonstrate cross-modal fusion without a large Transformer stack.
- **Frozen ASR:** classifier is the research contribution; ASR/TTS are assistive utilities.
