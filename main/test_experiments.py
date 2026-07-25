import torch
import traceback
from src.core.training import _build_model

# Mock configuration class
class MockConfig:
    def __init__(self, m_type):
        self.model_type = m_type
        self.loss_type = "bernoulli_gamma"
        self.emb_size = 64
        self.patch_size = 4
        self.num_layers = 2
        self.num_heads = 4
        self.dropout = 0.1
        self.training_dropout_value = 0.1
        self.group_norm_enable = False
        self.group_norm_num_groups = 8

def test_models():
    # Mock data shapes (matching typical downscaling: B=2, C=15, H=35, W=50 -> H_out=160, W_out=170)
    # Using smaller sizes for quick testing
    B, C, H, W = 2, 15, 30, 40
    H_out, W_out = 60, 70
    
    x_train = torch.randn(B, C, H, W)
    y_train = torch.randn(B, 3, H_out, W_out)
    
    models_to_test = [f"unet_exp{i}" for i in range(1, 11)] + [f"vit_exp{i}" for i in range(1, 9)]
    
    print(f"Testing {len(models_to_test)} experimental architectures on CPU...")
    print(f"Input shape: {x_train.shape} -> Output shape target: {y_train.shape}")
    print("-" * 50)
    
    passed = 0
    failed = []
    
    for m_type in models_to_test:
        cfg = MockConfig(m_type)
        try:
            model = _build_model(cfg, x_train, y_train)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            optimizer.zero_grad()
            out = model(x_train)
            
            # Check loss_type expectation
            if out.shape != y_train.shape:
                raise ValueError(f"Shape mismatch! Expected {y_train.shape}, got {out.shape}")
                
            # Fake loss backward
            loss = out.sum()
            loss.backward()
            optimizer.step()
                
            print(f"✅ {m_type: <12} PASSED (Output: {out.shape})")
            passed += 1
        except Exception as e:
            print(f"❌ {m_type: <12} FAILED: {str(e)}")
            failed.append((m_type, traceback.format_exc()))
            
    print("-" * 50)
    print(f"Summary: {passed}/{len(models_to_test)} passed.")
    
    if failed:
        print("\n--- Error Details ---")
        for m, trace in failed:
            print(f"\n{m}:")
            print(trace)

if __name__ == "__main__":
    test_models()
