# glowgenius
Projects powered by Thinking Machines!
Current Project: Supervised training loop with NYC Data using LORA
Setup/API Info: https://tinker-docs.thinkingmachines.ai/

#Project overview

This project creates a loop using Lora fine tuning methods and supervised learning to adapt a base language model using tinker API to answer questions about New York City. 

The goal is to understand:
1. How supervised learning affects model training
2. How LoRA fine tuning is used to amplify results from the base LLM
3. Benefits of the above approaches


Review:
1. Supervised learning is a machine learning method that is built off input/output data. The underlying model gets fed a pair of input/output data to learn from based off pattern recognition and the relationshio between a set on input/output data. Supervised learning is great for companies to train models off specfic use cases like identifying spam or predicting product purchases using historical data. 

2. LORA Fine tuning: Fine tuning uses a pre-trained larger LLM (base model) and trains the base model on a task using a specific dataset. LORA (low rank adaption),  enabled by the thinking machines API, introduces low rank adapters into the model so there are fewer parameters to train and the base model weights stay frozen. 

Adapters: A smaller set of paramaters that introduced onto on the large base model used for training
Amount of trainable paramaters used on this project: 494M	adpater weightd
WeightsL
LORA Learning Rank:
LORA Learning Rate:
Tokens:




3. This approach significantly reduces compute power when working with base model LLMS wihtout scarificing results and requires far feeer trainable paraterms, This makes training much more efficient, especially for small to medium datasets, while still allowing the model to adapt effectively to new tasks.

   

Summary of process: 


Thinking Machines Blog on LORA FT: https://thinkingmachines.ai/blog/lora/
