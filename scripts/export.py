import os
import sys
import json
import torch

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Assurer que 'src' est dans le chemin d'importation pour pouvoir unpickle modele_final
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.model import WishAI_BPE, ConfigModele

def _load_model(pt_file, model_dir):
    print("Chargement du modele PyTorch...")
    modele_ou_dict = torch.load(pt_file, map_location="cpu", weights_only=False)
    
    if isinstance(modele_ou_dict, dict):
        print("  Format checkpoint detecte. Reconstruction de l'architecture...")
        log_file = os.path.join(model_dir, "log_active.json")
        if not os.path.exists(log_file):
            raise Exception(f"Impossible de trouver {log_file} pour lire les hyperparametres.")
            
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
            
        hp = log_data.get("hyperparams", {})
        cfg = ConfigModele(
            vocab_size=hp.get("taille_vocab", 4000),
            n_embd=hp.get("n_embd", 64),
            n_head=hp.get("n_head", 4),
            n_layer=hp.get("n_layer", 4),
            block_size=hp.get("block_size", 128),
            dropout=0.0
        )
        modele = WishAI_BPE(cfg)
        
        state_dict = modele_ou_dict.get("modele_state", modele_ou_dict)
        clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        modele.load_state_dict(clean_dict)
        return modele
    else:
        return modele_ou_dict

def exporter_onnx(model_dir):
    """Exporte le modele vers ONNX."""
    print("\n--- Exportation vers ONNX ---")
    fichiers_pt = [
        os.path.join(model_dir, "modele.pt"),
        os.path.join(model_dir, "best_model.pt"),
        os.path.join(model_dir, "checkpoint.pt")
    ]
    pt_file = next((f for f in fichiers_pt if os.path.exists(f)), None)
    
    if not pt_file:
        print(f"[ERREUR] Aucun fichier modele.pt, best_model.pt ou checkpoint.pt trouve dans {model_dir}")
        return

    import onnx
    try:
        modele = _load_model(pt_file, model_dir)
    except Exception as e:
        print(f"[ERREUR] Chargement echoue : {e}")
        return
        
    modele.eval()
    
    cfg = modele.cfg
    onnx_file = os.path.join(model_dir, "modele.onnx")

    dummy_input = torch.randint(0, cfg.vocab_size, (1, cfg.block_size), dtype=torch.long)
    
    print(f"Exportation ONNX en cours (cela peut prendre du temps)...")
    try:
        torch.onnx.export(
            modele,
            dummy_input,
            onnx_file,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"  [OK] Modele exporte avec succes : {onnx_file}")
    except Exception as e:
        print(f"  [ERREUR] Exportation ONNX echouee : {e}")

def exporter_gguf(model_dir):
    """Exporte le modele vers GGUF."""
    print("\n--- Exportation vers GGUF ---")
    fichiers_pt = [
        os.path.join(model_dir, "modele.pt"),
        os.path.join(model_dir, "best_model.pt"),
        os.path.join(model_dir, "checkpoint.pt")
    ]
    pt_file = next((f for f in fichiers_pt if os.path.exists(f)), None)
    
    if not pt_file:
        print(f"[ERREUR] Aucun fichier modele.pt, best_model.pt ou checkpoint.pt trouve dans {model_dir}")
        return

    try:
        import gguf
    except ImportError:
        print("[ERREUR] Le module 'gguf' n'est pas installe. Lance: ./wish repair")
        return

    try:
        modele = _load_model(pt_file, model_dir)
    except Exception as e:
        print(f"[ERREUR] Chargement echoue : {e}")
        return
        
    cfg = modele.cfg
    state_dict = modele.state_dict()
    
    gguf_file = os.path.join(model_dir, "modele.gguf")
    
    mapping = {
        "token_embedding.weight": "token_embd.weight",
        "norm_finale.weight": "output_norm.weight",
        "tete_lm.weight": "output.weight"
    }
    
    for i in range(cfg.n_layer):
        mapping[f"blocs.{i}.attention.wq.weight"] = f"blk.{i}.attn_q.weight"
        mapping[f"blocs.{i}.attention.wk.weight"] = f"blk.{i}.attn_k.weight"
        mapping[f"blocs.{i}.attention.wv.weight"] = f"blk.{i}.attn_v.weight"
        mapping[f"blocs.{i}.attention.proj_sortie.weight"] = f"blk.{i}.attn_output.weight"
        mapping[f"blocs.{i}.ffn.w1.weight"] = f"blk.{i}.ffn_gate.weight"
        mapping[f"blocs.{i}.ffn.w2.weight"] = f"blk.{i}.ffn_up.weight"
        mapping[f"blocs.{i}.ffn.w3.weight"] = f"blk.{i}.ffn_down.weight"
        mapping[f"blocs.{i}.norm1.weight"] = f"blk.{i}.attn_norm.weight"
        mapping[f"blocs.{i}.norm2.weight"] = f"blk.{i}.ffn_norm.weight"

    print(f"Exportation GGUF en cours...")
    
    writer = gguf.GGUFWriter(gguf_file, "llama")
    
    writer.add_name("WishAI-Model")
    writer.add_context_length(cfg.block_size)
    writer.add_embedding_length(cfg.n_embd)
    writer.add_block_count(cfg.n_layer)
    # Dans WishAI, la taille cachee est cfg.n_embd // cfg.n_head ?? Non, dans model.py c'est une valeur qu'on calcule
    # Regardons FeedForward dans model.py:
    # La taille de projection est calculee a la creation de w1
    # On va calculer la taille reelle en regardant la shape de w1
    w1_shape = state_dict["blocs.0.ffn.w1.weight"].shape
    ff_dim = w1_shape[0] # w1 est (ff_dim, n_embd)
    writer.add_feed_forward_length(ff_dim)
    
    writer.add_head_count(cfg.n_head)
    writer.add_head_count_kv(cfg.n_head) # Pas de GQA par defaut dans WishAI
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    for name, tensor in state_dict.items():
        if name in mapping:
            gguf_name = mapping[name]
        elif "masque_causal" in name:
            continue
        else:
            continue
            
        tensor_np = tensor.float().detach().numpy() # Force float32
        writer.add_tensor(gguf_name, tensor_np)

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"  [OK] Modele exporte avec succes : {gguf_file}")
    except Exception as e:
        print(f"  [ERREUR] Exportation GGUF echouee : {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: ./wish export <nom_du_modele>")
        sys.exit(1)
        
    model_name = sys.argv[1]
    model_dir = os.path.join(ROOT, "model", model_name)
    
    if not os.path.exists(model_dir):
        print(f"[ERREUR] Le modele '{model_name}' n'existe pas dans le dossier model/")
        sys.exit(1)
        
    print(f"=== Exportation du modele : {model_name} ===")
    exporter_onnx(model_dir)
    exporter_gguf(model_dir)
    print("\nTermine !")

if __name__ == "__main__":
    main()
