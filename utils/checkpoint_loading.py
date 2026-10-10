"""Strict post-training load keeps every inherited action/tactile parameter."""


def load_resume_weights(model, state_dict, *, strict=False, report=print):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    expected = model.state_dict()
    mismatches = [key for key, value in state_dict.items() if key in expected and value.shape != expected[key].shape]
    missing = sorted(expected.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected.keys())
    if strict and (mismatches or missing or unexpected):
        raise ValueError(
            f"Incompatible post-training checkpoint: shape={mismatches}, missing={missing}, unexpected={unexpected}"
        )
    filtered = {key: value for key, value in state_dict.items() if key not in mismatches}
    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    report(f"Resume load: skipped_shapes={mismatches}, missing={list(missing)}, unexpected={list(unexpected)}")
    return missing, unexpected
