import torch

# 1. Versione di PyTorch installata
print("PyTorch:", torch.__version__)

# 2. PyTorch riesce a vedere la GPU?
print("CUDA disponibile:", torch.cuda.is_available())

# 3. Nome della GPU (la numero 0, perché ne hai una sola)
print("GPU:", torch.cuda.get_device_name(0))

# 4. Un calcolo vero sulla GPU
a = torch.randn(4096, 4096, device="cuda")  # matrice 4096x4096 di numeri casuali, creata direttamente sulla GPU
b = torch.randn(4096, 4096, device="cuda")  # un'altra matrice uguale
c = a @ b  # prodotto righe-per-colonne di a e b
print("Forma del risultato:", c.shape)
print("Dove vive c:", c.device)
