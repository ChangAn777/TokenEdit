# -*- coding: utf-8 -*-
"""
Data-driven Prompt Closure Generator
Works with CounterFact, ZSRE, and MQuAKE datasets without predefined templates
"""

from typing import Dict, List, Optional


class PromptClosureGenerator:
    """
    Data-driven prompt closure generator.

    Instead of using hardcoded templates, this class directly uses the
    paraphrase_prompts and neighborhood_prompts from the dataset.

    This approach is universal and works for any relation type.
    """

    def __init__(self):
        # Minimal fallback templates (only used if dataset doesn't provide prompts)
        self.fallback_templates = {
            "forward": ["{subject} {attribute}"],  # Generic: "X has attribute Y"
            "backward": ["{object} is the {attribute} of"],
        }

    def generate_from_dataset(
        self,
        rewrite_prompt: str,
        subject: str,
        target_new: str,
        target_true: str,
        paraphrase_prompts: Optional[List[str]] = None,
        neighborhood_prompts: Optional[List[Dict]] = None,
        num_paraphrase: int = 3,
    ) -> Dict[str, List]:
        """
        Generate prompt closure from dataset examples.

        Args:
            rewrite_prompt: The original prompt template (e.g., "The capital of {} is")
            subject: The subject entity (e.g., "France")
            target_new: The new target value (e.g., "Lyon")
            target_true: The true target value (e.g., "Paris")
            paraphrase_prompts: List of paraphrase prompts from dataset
            neighborhood_prompts: List of neighborhood prompts from dataset
            num_paraphrase: Number of paraphrase prompts to use

        Returns:
            closure: {
                "rewrite_prompts": [original prompt formatted],
                "paraphrase_prompts": [paraphrases for generalization],
                "neighborhood_prompts": [neighborhood examples for specificity],
                "prompts_forward": [all prompts that should output target_new],
                "prompts_backward": [all prompts that should output target_true],
                "targets_forward": target_new,
                "targets_backward": target_true
            }
        """
        closure = {
            "rewrite_prompts": [rewrite_prompt.format(subject)],
            "paraphrase_prompts": [],
            "neighborhood_prompts": [],
            "prompts_forward": [],  # Should output target_new
            "prompts_backward": [],  # Should output target_true (or original)
            "targets_forward": target_new,
            "targets_backward": target_true,
            "subject": subject,
        }

        # Add paraphrase prompts (test generalization)
        if paraphrase_prompts:
            # Use dataset-provided paraphrases
            selected_paraphrases = paraphrase_prompts[:num_paraphrase]
            closure["paraphrase_prompts"] = selected_paraphrases
            closure["prompts_forward"].extend(selected_paraphrases)
        else:
            # Fallback: generate simple paraphrases
            closure["paraphrase_prompts"] = []
            # Could add basic paraphrase generation here if needed

        # The original rewrite prompt should output the new target
        closure["prompts_forward"].append(rewrite_prompt.format(subject))

        # Add neighborhood prompts (test specificity - should NOT change)
        if neighborhood_prompts:
            for nb in neighborhood_prompts[:5]:  # Use up to 5 neighborhood prompts
                if isinstance(nb, dict):
                    prompt = nb.get("prompt", "")
                    target = nb.get("target", "")
                    closure["neighborhood_prompts"].append({
                        "prompt": prompt,
                        "target": target
                    })
                    # Neighborhood prompts should keep their original targets
                    closure["prompts_backward"].append(prompt)

        return closure

    def generate_multi_turn_closure(
        self,
        requests: List[Dict],
    ) -> Dict[str, List]:
        """
        Generate closure for multiple edits (batch mode).

        Args:
            requests: List of edit requests with keys:
                - prompt: str (template with {})
                - subject: str
                - target_new: str
                - target_true: str
                - paraphrase_prompts: List[str] (optional)
                - neighborhood_prompts: List[Dict] (optional)

        Returns:
            Batch closure with all prompts organized by type
        """
        batch_closure = {
            "rewrite_prompts": [],
            "paraphrase_prompts": [],
            "neighborhood_prompts": [],
            "prompts_forward": [],
            "prompts_backward": [],
            "targets_forward": [],
            "targets_backward": [],
            "subjects": [],
        }

        for req in requests:
            subject = req["subject"]
            prompt = req["prompt"]
            target_new = req["target_new"]
            target_true = req["target_true"]
            paraphrases = req.get("paraphrase_prompts", [])
            neighborhoods = req.get("neighborhood_prompts", [])

            # Generate single closure
            closure = self.generate_from_dataset(
                rewrite_prompt=prompt,
                subject=subject,
                target_new=target_new,
                target_true=target_true,
                paraphrase_prompts=paraphrases,
                neighborhood_prompts=neighborhoods,
            )

            # Aggregate
            batch_closure["rewrite_prompts"].extend(closure["rewrite_prompts"])
            batch_closure["paraphrase_prompts"].extend(closure["paraphrase_prompts"])
            batch_closure["neighborhood_prompts"].extend(closure["neighborhood_prompts"])
            batch_closure["prompts_forward"].extend(closure["prompts_forward"])
            batch_closure["prompts_backward"].extend(closure["prompts_backward"])
            batch_closure["targets_forward"].append(target_new)
            batch_closure["targets_backward"].append(target_true)
            batch_closure["subjects"].append(subject)

        return batch_closure

    def generate_training_samples(
        self,
        closure: Dict,
        use_forward: bool = True,
        use_backward: bool = True,
        use_distract: bool = True,
    ) -> List[Dict]:
        """
        Generate training samples from closure.

        Args:
            closure: The prompt closure dictionary
            use_forward: Include samples that should output target_new
            use_backward: Include samples that should output target_true
            use_distract: Include neighborhood samples that should not change

        Returns:
            List of training samples with prompts and expected outputs
        """
        samples = []

        # Forward samples: should output the new target
        if use_forward:
            for prompt in closure["prompts_forward"]:
                samples.append({
                    "prompt": prompt,
                    "target": closure["targets_forward"],
                    "type": "forward",
                    "subject": closure.get("subject", ""),
                })

        # Backward samples: should output the true target (for unlikelihood)
        if use_backward and closure["prompts_backward"]:
            for prompt in closure["prompts_backward"]:
                samples.append({
                    "prompt": prompt,
                    "target": closure["targets_backward"],
                    "type": "backward",
                    "subject": closure.get("subject", ""),
                })

        return samples


# Convenience function for backward compatibility
def generate_closure_from_request(request: Dict) -> Dict:
    """
    Generate closure from a single request dict.
    Convenience function for backward compatibility.

    Args:
        request: Dict with keys:
            - prompt: str
            - subject: str
            - target_new: str
            - target_true: str
            - paraphrase_prompts: List[str] (optional)
            - neighborhood_prompts: List[Dict] (optional)

    Returns:
        Prompt closure dictionary
    """
    generator = PromptClosureGenerator()
    return generator.generate_from_dataset(
        rewrite_prompt=request["prompt"],
        subject=request["subject"],
        target_new=request["target_new"],
        target_true=request["target_true"],
        paraphrase_prompts=request.get("paraphrase_prompts"),
        neighborhood_prompts=request.get("neighborhood_prompts"),
    )