# Predicting IPC-Technology Code Popularity based on Text-Embeddings and Graph Structure

The goal of this project is to predict the popularity of different technologies by forecasting how often specific International Patent Classification (IPC) codes will appear in patents over the following year.
IPC codes are a global system that is used to categorize patents into specific technical fields. For example, all codes starting with "A" are for human necessities while all codes starting with "H" are for electronics \cite{wipo2026ipc}. The codes are then further hierarchically classified into a total of around 70,000.
Having the ability to predict the future popularity of IPC codes can have many benefits for companies, investors, and governments. 
The gained insight can lead to more stable supply chains, better trade deals, and more strategic R\&D investments, as it helps them anticipate where the market is moving.
This project is based on the work of \textcite{aroyehun2024anticipating}. With the help of LLMs, they created text embeddings for many IPC codes. That is, each IPC code is placed in a multidimensional space, based on what patents it is used in and how these patents are described.
We can now use these embeddings for our project with the idea that the placement and movement of an IPC code in the embedding space can somehow be an indicator of its future popularity.
To explore this, I am training three different models: a standard Multilayer Perceptron (MLP) just using the embeddings and some scalar values, a Graph Neural Network (GNN) that maps the relationships between codes based on co-occurrences, and a Temporal GNN that combines a GNN with a Gated Recurrent Unit (GRU) to track how those technological relationships evolve over time.
