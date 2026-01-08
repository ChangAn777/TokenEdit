"""Prompt闭包生成器"""

from typing import Dict, List

class PromptClosureGenerator:
    """
    Prompt闭包生成器
    
    为每个编辑生成四类训练样本:
    1. 正向问法: "X的Y是?"
    2. 反向问法: "Z是什么的Y?"
    3. 判断问法: "X的Y是Z吗?"
    4. 干扰问法: "X的W是?" (无关问题)
    """
    
    def __init__(self):
        # 关系模板库
        self.templates = {
            "capital": {
                "forward": [
                    "The capital of {subject} is",
                    "{subject}的首都是",
                    "What is the capital of {subject}?",
                ],
                "backward": [
                    "{object} is the capital of",
                    "{object}是哪个国家的首都?",
                    "Which country has {object} as its capital?",
                ],
                "judge": [
                    "Is {object} the capital of {subject}?",
                    "{subject}的首都是{object}吗?",
                ],
                "distract": [
                    "The population of {subject} is",
                    "{subject}的人口是多少?",
                    "What language is spoken in {subject}?",
                ]
            },
            "president": {
                "forward": [
                    "The president of {subject} is",
                    "{subject}的总统是",
                ],
                "backward": [
                    "{object} is the president of",
                    "{object}是哪个国家的总统?",
                ],
                "judge": [
                    "Is {object} the president of {subject}?",
                ],
                "distract": [
                    "The capital of {subject} is",
                    "{subject}的GDP是多少?",
                ]
            },
            # 可扩展更多关系
        }
    
    def generate(
        self, 
        subject: str, 
        relation: str, 
        new_object: str, 
        old_object: str
    ) -> Dict[str, List[str]]:
        """
        生成Prompt闭包
        
        Args:
            subject: 主体
            relation: 关系
            new_object: 新对象
            old_object: 旧对象
        
        Returns:
            closure: {
                "forward": [prompts],
                "backward": [prompts],
                "judge": [prompts],
                "distract": [prompts],
                "targets": {...}
            }
        """
        if relation not in self.templates:
            # 使用通用模板
            relation = "capital"  # 默认
        
        templates = self.templates[relation]
        
        closure = {
            "forward": [
                t.format(subject=subject) for t in templates["forward"]
            ],
            "backward": [
                t.format(object=new_object) for t in templates["backward"]
            ],
            "judge": [
                t.format(subject=subject, object=new_object) 
                for t in templates["judge"]
            ],
            "distract": [
                t.format(subject=subject) for t in templates["distract"]
            ],
            "targets": {
                "forward": new_object,
                "backward": subject,
                "judge": "Yes",
                "distract": None  # 保持原状
            },
            "old_object": old_object  # 用于Unlikelihood Loss
        }
        
        return closure
