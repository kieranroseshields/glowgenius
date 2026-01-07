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
1. Supervised learning is a machine learning method that is built off input/output data. The underlying model gets fed a pair of input/output data to learn from based off pattern recognition and the relationship between a set of input/output data. Supervised learning is great for companies to train models for specific use cases like identifying spam or predicting product purchases using historical data. 

2. LORA Fine tuning: Fine tuning uses a pre-trained larger LLM (base model) and trains the base model on a task using a specific dataset. LORA (low rank adaption),  enabled by the thinking machines API, introduces low rank adapters into the model so there are fewer parameters to train and the base model weights stay frozen. 

Tokens : Numbers that translate the text and drive learning

Base Weights: Define how Base model already thinks - frozen during LORA

LORA Adapters: Inserted weights into the layers

LORA Learning Rank: capacity of learning - smaller rank more subtle changes to the data, higher rank larger more expressive changes

LORA Learning Rate: how fast the adapters change, lower the rate the safer the learning


3. The LORA approach significantly reduces compute power when working with base model LLMs without sacrificing results. LORA requires far fewer trainable parameters, this makes training much more efficient, especially for small to medium datasets. When adding supervised learning with LORA fine tuning, you can create a tailored model using input/output data while using low rank adapters to use base models efficiently while still gaining the stability and general knowledge of the base model. 
   



Thinking Machines Blog on LORA FT: https://thinkingmachines.ai/blog/lora/
