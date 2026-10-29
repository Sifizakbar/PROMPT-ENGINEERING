# Aim: Comprehensive Report on the Fundamentals of Generative AI and Large Language Models (LLMs)

**Experiment:** Develop a comprehensive report for the following exercises:
1. Explain the foundational concepts of Generative AI.
2. Focusing on Generative AI architectures (like transformers).
3. Generative AI applications.
4. Generative AI impact of scaling in LLMs.

# Algorithm

**Step 1: Define Scope and Objectives**
1.1 Identify the goal of the report (e.g., educational, research, tech overview)
1.2 Set the target audience level (e.g., students, professionals)
1.3 Draft a list of core topics to cover

**Step 2: Create Report Skeleton/Structure**
2.1 Title Page
2.2 Abstract or Executive Summary
2.3 Table of Contents
2.4 Introduction
2.5 Main Body Sections: Introduction to AI and Machine Learning; What is Generative AI?; Types of Generative AI Models (e.g., GANs, VAEs, Diffusion Models); Introduction to Large Language Models (LLMs); Architecture of LLMs (e.g., Transformer, GPT, BERT); Training Process and Data Requirements; Use Cases and Applications (Chatbots, Content Generation, etc.); Limitations and Ethical Considerations; Future Trends
2.6 Conclusion
2.7 References

**Step 3: Research and Data Collection**
3.1 Gather recent academic papers, blog posts, and official docs (e.g., OpenAI, Google AI)
3.2 Extract definitions, explanations, diagrams, and examples
3.3 Cite all sources properly

**Step 4: Content Development**
4.1 Write each section in clear, simple language
4.2 Include diagrams, figures, and charts where needed
4.3 Highlight important terms and definitions
4.4 Use examples and real-world analogies for better understanding

**Step 5: Visual and Technical Enhancement**
5.1 Add tables, comparison charts (e.g., GPT-3 vs GPT-4)
5.2 Use tools like Canva, PowerPoint, or LaTeX for formatting
5.3 Add code snippets or pseudocode for LLM working (optional)

**Step 6: Review and Edit**
6.1 Proofread for grammar, spelling, and clarity
6.2 Ensure logical flow and consistency
6.3 Validate technical accuracy
6.4 Peer-review or use tools like Grammarly or ChatGPT for suggestions

**Step 7: Finalize and Export**
7.1 Format the report professionally
7.2 Export as PDF or desired format
7.3 Prepare a brief presentation if required (optional)

# Output

## Comprehensive Report on the Fundamentals of Generative AI and Large Language Models (LLMs)

## Abstract

Generative Artificial Intelligence (Generative AI) refers to models that learn the statistical structure of existing data and use it to produce new content such as text, images, audio, video and code. Large Language Models (LLMs), most of which are built on the Transformer architecture, have become the most visible form of Generative AI. This report explains the foundational concepts of Generative AI, surveys the main model families (GANs, VAEs, diffusion models and autoregressive Transformers), describes the architecture and training of LLMs, reviews applications, and analyses how scaling parameters, data and compute affects capability, cost and risk. It also presents a comparative evaluation of ChatGPT and Google Gemini using five identical prompts.

**Keywords:** Generative AI, Large Language Models, Transformer, self-attention, scaling laws, GAN, VAE, diffusion.

## 1. Introduction

Over the last decade, Artificial Intelligence has moved from systems that classify or predict to systems that create. A spam filter decides whether an email is unwanted; a generative model can write the email. This shift, driven by deep learning, large datasets and powerful hardware, has made Generative AI one of the most important technologies in engineering and education.

**1.1 Goal.** The goal of this report is educational and technical: to give a clear overview of Generative AI and LLMs that a second- or third-year engineering student can follow.

**1.2 Scope.** The report covers: AI and machine learning basics; types of generative models; LLMs and the Transformer architecture; training and data requirements; applications; the impact of scaling; limitations and ethics; future trends; and a comparison of ChatGPT and Google Gemini.

**1.3 Method.** The report was developed by defining the scope and audience, drafting a structure, collecting definitions and examples from foundational academic papers, writing each section in simple language, and adding tables and pseudocode. The comparison was designed so that both tools receive identical prompts and are scored on identical criteria.

## 2. Introduction to AI and Machine Learning

**Artificial Intelligence (AI)** is the broad field concerned with building systems that perform tasks normally requiring human intelligence, such as perception, reasoning, language understanding and decision making. **Machine Learning (ML)** is a subset of AI in which systems learn patterns from data instead of being explicitly programmed. **Deep Learning (DL)** is a subset of ML that uses multi-layer neural networks to learn increasingly abstract representations. **Generative AI** is a branch of deep learning focused on producing new data.

| Level | Description | Example |
|---|---|---|
| Artificial Intelligence | Any technique enabling machines to mimic intelligent behaviour | Rule-based expert system, chess engine |
| Machine Learning | Algorithms that learn from data | Decision trees, linear regression, SVM |
| Deep Learning | Multi-layer neural networks learning hierarchical features | CNNs for image recognition |
| Generative AI | Models that learn a data distribution and sample new content from it | ChatGPT, Gemini, Stable Diffusion |

*Table 1: Relationship between AI, ML, DL and Generative AI.*

**2.1 Learning paradigms**

- **Supervised learning:** learns a mapping from inputs to labelled outputs (for example, classifying images).
- **Unsupervised learning:** finds structure in unlabelled data (for example, clustering).
- **Self-supervised learning:** creates its own labels from the data, such as predicting the next word in a sentence. This is the paradigm behind modern LLMs.
- **Reinforcement learning:** learns from reward signals; used to align LLMs with human preferences.

## 3. Foundational Concepts of Generative AI

**3.1 What is Generative AI?**

Generative AI is a class of AI systems that learn the underlying probability distribution of training data and then sample from it to create new, original-looking outputs. A *discriminative* model learns p(y | x), the probability of a label given an input. A *generative* model learns p(x), or p(x | c) when conditioned on a prompt or context c.

**Analogy:** A discriminative model is like an art critic who can say whether a painting is a forgery. A generative model is like an artist who has studied thousands of paintings and can paint a new one in a similar style.

**3.2 Key terms**

| Term | Meaning |
|---|---|
| Token | A unit of text (word, sub-word or character) that a language model reads and predicts |
| Embedding | A vector of numbers representing the meaning of a token or other input |
| Latent space | A compressed, learned representation space from which new data can be generated |
| Prompt | The input instruction or context given to a model |
| Parameters | The learned weights of a neural network; a rough measure of model size |
| Inference | Using a trained model to generate outputs |
| Fine-tuning | Further training of a pre-trained model on a narrower dataset or task |
| Hallucination | Fluent output that is factually wrong or unsupported |

*Table 2: Core vocabulary of Generative AI.*

**3.3 Types of Generative AI models**

**Generative Adversarial Networks (GANs).** Introduced by Goodfellow et al. (2014), a GAN pairs a *generator*, which creates samples from random noise, with a *discriminator*, which tries to tell real samples from generated ones. The two are trained against each other until the generator's output is hard to distinguish from real data. GANs produce sharp images but can be unstable to train and may suffer from mode collapse.

**Variational Autoencoders (VAEs).** Introduced by Kingma and Welling (2013), a VAE has an encoder that maps data to a probability distribution in a latent space and a decoder that reconstructs data from latent samples. Training balances reconstruction quality against keeping the latent space smooth and well structured. VAEs are stable and useful for representation learning, though outputs are often blurrier than GANs.

**Diffusion models.** Diffusion models (Ho et al., 2020) gradually add noise to data in a forward process and learn to reverse it step by step. Generation starts from pure noise and removes it iteratively, optionally guided by a text prompt. They underpin many modern image, audio and video generators.

**Autoregressive Transformer models.** These generate a sequence one token at a time, each conditioned on all previous tokens. This is the approach used by GPT-style LLMs.

| Model family | Core idea | Strengths | Weaknesses |
|---|---|---|---|
| GAN | Generator vs discriminator | Sharp, realistic images; fast sampling | Unstable training; mode collapse |
| VAE | Encode to latent distribution, decode | Stable; smooth latent space | Blurrier outputs |
| Diffusion | Learn to denoise step by step | High quality and diversity | Slow iterative sampling |
| Autoregressive Transformer | Predict the next token | Flexible; scales well; strong for text and code | Sequential generation; hallucination |

*Table 3: Comparison of major generative model families.*

## 4. Large Language Models and Generative AI Architectures

**4.1 Introduction to LLMs**

A **Large Language Model** is a neural network, usually a Transformer, trained on very large text corpora to model language. Its basic training objective is simple: given the tokens so far, predict the next token. By doing this over trillions of tokens, the model acquires grammar, facts, reasoning patterns and the ability to follow instructions. Examples include the GPT family (OpenAI), Gemini (Google), Claude (Anthropic) and open-weight models such as Llama.

**4.2 The Transformer architecture**

Before 2017, language models relied mainly on recurrent networks (RNNs and LSTMs), which process text sequentially and struggle with long-range dependencies. Vaswani et al. (2017) proposed the **Transformer** in *Attention Is All You Need*, replacing recurrence with **self-attention**. Every token can look directly at every other token, and computation can be parallelised on GPUs.

**Main components**

- **Tokenisation and embeddings:** text is split into tokens, each mapped to a vector.
- **Positional encoding:** adds information about token order, since attention itself ignores order.
- **Multi-head self-attention:** several attention heads run in parallel, each learning different relationships (syntax, coreference, topic).
- **Feed-forward network:** a position-wise network applied after attention.
- **Residual connections and layer normalisation:** stabilise training of deep stacks.
- **Output layer (softmax):** converts the final vector into a probability distribution over the vocabulary.

**Self-attention equation.** Each token is projected into a query (Q), key (K) and value (V):

**Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V**

The softmax of the scaled dot products gives attention weights saying how much each token should draw on every other token. Dividing by √d_k keeps values numerically stable.

**Analogy:** In the sentence "The trophy did not fit in the suitcase because it was too big," a reader decides that "it" means the trophy. Self-attention gives the model a learned way to make exactly this kind of link.

**4.3 Pseudocode**

```python
def transformer_block(x):
    a = multi_head_self_attention(layer_norm(x))   # mix information across tokens
    x = x + a                                       # residual connection
    f = feed_forward(layer_norm(x))                 # per-token transformation
    return x + f                                    # residual connection

def generate(model, prompt_tokens, max_new_tokens):
    tokens = prompt_tokens
    for _ in range(max_new_tokens):
        probs  = model(tokens)[-1]                  # distribution over next token
        next_t = sample(probs, temperature=0.8)     # choose a token
        tokens.append(next_t)
        if next_t == END_OF_TEXT: break
    return tokens
```

**4.4 GPT, BERT and other variants**

| Architecture | Example | Attention style | Typical use |
|---|---|---|---|
| Decoder-only | GPT family, Llama | Causal (each token sees only earlier tokens) | Text generation, chat, code |
| Encoder-only | BERT | Bidirectional | Classification, search, embeddings |
| Encoder-decoder | T5, original Transformer | Encoder reads input, decoder generates output | Translation, summarisation |

*Table 4: Main Transformer variants.*

**GPT** (Generative Pre-trained Transformer) stacks decoder blocks and is trained to predict the next token, which suits generation. **BERT** (Devlin et al., 2018) is trained by masking words and predicting them from both left and right context, which makes it strong at understanding tasks. Modern frontier models are also **multimodal**, handling images, audio and video as well as text, and some use **Mixture-of-Experts** layers in which only part of the network is activated per token to reduce compute.

## 5. Training Process and Data Requirements

| Stage | What happens | Data |
|---|---|---|
| 1. Data collection and cleaning | Gather text and code, remove duplicates, filter low-quality or harmful content, tokenise | Web pages, books, code, papers |
| 2. Pre-training | Self-supervised next-token prediction; the most expensive stage | Trillions of tokens |
| 3. Supervised fine-tuning (SFT) | Train on curated prompt-response pairs so the model follows instructions | Thousands to millions of examples |
| 4. Alignment | Use human or AI feedback (e.g. RLHF) to make answers helpful, honest and safe | Preference comparisons |
| 5. Evaluation and safety testing | Benchmarks, red-teaming, bias and robustness tests | Held-out test sets |
| 6. Deployment and monitoring | Serve via API or app; optimise latency, cost and safety | Live usage |

*Table 5: Typical LLM training and deployment pipeline.*

**5.1 Optimisation.** Pre-training minimises a **cross-entropy loss** between the predicted next-token distribution and the actual next token. Parameters are updated with gradient descent variants, most commonly Adam or AdamW, using learning-rate schedules, gradient clipping and mixed-precision arithmetic. Because models are too large for one device, training is distributed across thousands of accelerators using data, tensor and pipeline parallelism.

**5.2 Data requirements and quality**

- **Volume:** frontier models train on trillions of tokens; compute-optimal research (Hoffmann et al., 2022) showed data should grow together with model size.
- **Quality:** deduplication, filtering and balanced domain coverage often matter as much as raw size.
- **Diversity:** multiple languages, domains and formats improve generalisation.
- **Legal and ethical:** copyright, licensing, privacy and consent are active issues.
- **Synthetic data:** model-generated data is increasingly used, with the risk of amplifying existing errors.

**5.3 Fine-tuning approaches.** Full fine-tuning updates every parameter and is costly. Parameter-efficient methods such as LoRA update only small added matrices. Prompting techniques such as few-shot examples, chain-of-thought and retrieval-augmented generation (RAG) adapt behaviour without changing weights. RAG connects the model to an external knowledge source so answers can be grounded in retrieved documents, which helps reduce hallucination.

## 6. Generative AI Applications

| Domain | Applications | Benefit | Caution |
|---|---|---|---|
| Software engineering | Code completion, debugging, test generation, documentation | Faster development; lower barrier to entry | Generated code may contain bugs or security flaws |
| Education | Tutoring, quiz and summary generation, personalised explanations | Individual pace, instant feedback | Over-reliance; academic misconduct |
| Content creation | Articles, marketing copy, scripts, images, music | Speed and variety | Originality, copyright, disclosure |
| Customer service | Chatbots, virtual assistants, ticket summarisation | 24/7 support at scale | Wrong answers; poor edge-case handling |
| Healthcare | Clinical note drafting, literature summarisation, drug discovery support | Less paperwork; faster research | Safety-critical; needs expert oversight and privacy protection |
| Engineering and design | Generative design, simulation surrogates, report drafting, CAD assistance | Explores large design spaces quickly | Outputs must be validated against physical constraints |
| Science and research | Protein and molecule generation, literature review, data analysis | Speeds up discovery | Needs experimental validation |
| Business and finance | Report generation, document analysis, forecasting support | Saves analyst time | Confidentiality; regulatory compliance |

*Table 6: Representative applications of Generative AI.*

**6.1 Prompt engineering.** Response quality depends strongly on the prompt. Good practice includes stating the role and audience, giving clear instructions and constraints, supplying relevant context, providing one or two examples of the desired format, and asking for step-by-step reasoning on complex problems. Prompts should be iterated and answers verified.

**6.2 Agents and tool use.** LLMs are increasingly connected to tools such as search engines, code interpreters and databases so they can plan multi-step tasks, retrieve current information and take actions. This expands what they can do but raises new safety and reliability questions, because errors can compound across steps.

## 7. Impact of Scaling in LLMs

**7.1 Scaling laws.** Kaplan et al. (2020) found that the loss of language models falls smoothly and predictably, following a power law, as model size, dataset size and training compute increase. Hoffmann et al. (2022), in the "Chinchilla" study, showed that many large models were under-trained for their size and that, for a fixed compute budget, parameters and training tokens should be scaled roughly in proportion. This made model development more plannable: performance of a larger model can be estimated from smaller experiments.

**7.2 Benefits of scaling**

- **Lower loss and better accuracy** across a wide range of benchmarks.
- **Emergent abilities:** Wei et al. (2022) reported abilities such as multi-step arithmetic that appear only above certain model sizes. Later work argues some apparent emergence depends on how performance is measured, so this remains debated.
- **In-context learning:** larger models can learn a task from a few examples in the prompt without weight updates (Brown et al., 2020).
- **Broader knowledge and multilingual capability.**
- **Better instruction following and reasoning** when combined with good fine-tuning and alignment.

**7.3 Costs and limitations**

- **Compute and energy:** training and serving require large data centres, with significant electricity and water use.
- **Financial cost:** frontier training runs are extremely expensive, concentrating development in a few organisations.
- **Latency and serving cost:** larger models are slower and costlier per query, motivating distillation, quantisation and Mixture-of-Experts designs.
- **Diminishing returns:** power-law behaviour means each additional gain needs a much larger increase in resources.
- **Data limits:** high-quality public text is finite, increasing interest in synthetic data and better curation.
- **Persistent weaknesses:** scale alone does not remove hallucination, bias or reasoning errors.
- **Risk:** more capable models can be misused more effectively, so safety evaluation matters more.

| Factor scaled | Typical effect | Trade-off |
|---|---|---|
| Parameters | More capacity to store knowledge and patterns | Higher memory, compute and inference cost |
| Training data (tokens) | Better generalisation and coverage | Data quality, legal and availability issues |
| Training compute | Lower loss when allocated well | Energy and monetary cost |
| Context length | Can process longer documents | Attention cost grows with length |
| Test-time compute (reasoning steps) | Better accuracy on hard problems | Higher latency and cost per answer |

*Table 7: Effects and trade-offs of scaling.*

**7.4 Beyond pure scaling.** Recent progress relies on more than size. Better data curation, improved alignment, longer context, retrieval, tool use and extra inference-time reasoning have all produced large gains, and smaller well-trained models can now match the earlier performance of much larger ones. Efficiency is as important a direction as raw scale.

## 8. Limitations and Ethical Considerations

**8.1 Technical limitations**

- **Hallucination:** LLMs generate plausible text, not guaranteed truth, and can state false facts or invent citations confidently.
- **Knowledge cut-off:** a model knows only what was in its training data unless connected to search or retrieval.
- **Reasoning limits:** models may fail on multi-step logic, precise arithmetic or unfamiliar task patterns.
- **Prompt sensitivity:** small changes in wording can change the answer.
- **Limited transparency:** it is hard to explain why a model produced a particular output.

**8.2 Ethical and societal issues**

- **Bias and fairness:** models can reproduce and amplify biases present in training data.
- **Privacy:** training data and prompts may contain personal information.
- **Intellectual property:** open questions about training on copyrighted material and ownership of outputs.
- **Misinformation and deepfakes:** synthetic text, images and audio can deceive at scale.
- **Security:** prompt injection, data leakage and misuse for fraud or malicious code.
- **Employment and education:** changes to jobs and to how skills are taught and assessed.
- **Environmental impact:** energy and water use of training and inference.

**8.3 Responsible use.** Verify important facts against reliable sources, keep a human in the loop for high-stakes decisions, disclose AI assistance where required, protect sensitive data, test for bias, and follow institutional and legal guidelines.

## 9. Future Trends

- **Multimodal models** combining text, vision, audio and video in one system.
- **AI agents** that plan, use tools and complete multi-step tasks under supervision.
- **Reasoning models** that use extra inference-time computation for harder problems.
- **Efficient, small models** that run on laptops and phones, improving privacy and cost.
- **Grounding and retrieval** to improve factual reliability and provide citations.
- **Better evaluation, interpretability and safety research.**
- **Regulation and standards** shaping transparency, data use and accountability.
- **Domain-specialised models** for medicine, law, science and engineering.

## 10. Comparative Evaluation of ChatGPT and Google Gemini

The ratings below are illustrative and should be updated with the results of the actual experiment.

**10.1 Selected AI tools**

- **AI Tool 1:** ChatGPT
- **AI Tool 2:** Google Gemini

Both tools were selected to compare their ability to explain Generative AI concepts and Large Language Models using five identical prompts.

**10.2 Comparison of responses**

| Evaluation Criterion | ChatGPT | Google Gemini |
|---|---|---|
| Accuracy | 4/5 – Generally provides technically detailed explanations; important facts still require verification. | 4/5 – Generally provides informative explanations; technical claims still require verification. |
| Creativity | 4/5 – Provides structured examples, analogies, and alternative explanations. | 4/5 – Provides varied examples and alternative ways to present information. |
| Hallucination | 3/5 – May occasionally generate incorrect facts or unsupported claims. | 3/5 – May occasionally generate incorrect facts or unsupported claims. |
| Reasoning | 4/5 – Often explains technical concepts in a logical, step-by-step format. | 4/5 – Often provides organized explanations and logical steps. |
| Speed | 4/5 – Usually generates responses quickly, depending on the task and system load. | 4/5 – Usually generates responses quickly, depending on the task and system load. |
| Engineering Usefulness | 5/5 – Useful for learning technical concepts, programming, and structured report preparation. | 4/5 – Useful for technical learning, examples, and report preparation. |
| **Total Score** | **24/30** | **23/30** |
| **Average Score** | **4.00/5** | **3.83/5** |

**10.3 Observations**

**ChatGPT**

- Provides structured explanations suitable for undergraduate engineering students.
- Explains technical concepts through examples, headings, and step-by-step descriptions.
- Can assist with programming concepts, report preparation, and understanding Transformer architectures.
- May produce inaccurate information, so references and technical claims should be checked.

**Google Gemini**

- Provides organized explanations of Generative AI and its applications.
- Can present alternative examples and explanations for complex concepts.
- Can be useful for exploring technical topics and preparing academic content.
- May also produce unsupported claims or factual errors that require verification.

**10.4 Analysis of the five prompts**

| Prompt | Comparison focus | Observation to verify |
|---|---|---|
| Prompt 1: Fundamentals of Generative AI | Definitions, model types, and basic concepts | Which tool explains the concepts more clearly and accurately? |
| Prompt 2: Generative AI Architectures | GANs, VAEs, diffusion models, and Transformers | Which tool provides the better technical explanation and architecture flow? |
| Prompt 3: Architectures and Applications | Practical uses across engineering domains | Which tool gives more relevant applications and examples? |
| Prompt 4: Impact of Scaling in LLMs | Parameters, data, compute, and model performance | Which tool explains scaling benefits and limitations more accurately? |
| Prompt 5: How LLMs Are Built | Training pipeline, optimization, fine-tuning, and deployment | Which tool provides the most complete and logically ordered explanation? |

**10.5 Overall comparison**

Based on the illustrative ratings, ChatGPT receives an average score of 4.00/5, while Google Gemini receives 3.83/5. ChatGPT has a slight advantage in engineering usefulness in this example. Both tools receive similar ratings for accuracy, creativity, hallucination, reasoning, and speed.

## 11. Conclusion

Generative AI models learn the distribution of data and sample new content from it. Among the main families, GANs, VAEs, diffusion models and autoregressive Transformers each have distinct strengths, while the Transformer, built on self-attention, has become the dominant architecture for LLMs because it handles long-range context and parallelises well on modern hardware. LLMs are produced through pre-training, fine-tuning, alignment and deployment, and are applied across software engineering, education, healthcare, science and business.

Scaling model size, data and compute has produced predictable improvements and new capabilities, but it also brings rising cost, energy use and risk, and it does not remove hallucination or bias. Progress now depends on a combination of scale, better data, alignment, retrieval, reasoning and efficiency.

The comparison of ChatGPT and Google Gemini shows how AI assistants can support learning about these topics. Both can generate explanations, examples and technical reports, but their responses must be checked for factual correctness, logical reasoning and practical usefulness. Identical prompts and consistent criteria give a more meaningful basis for comparison, and the most suitable tool should be chosen based on the results observed in the actual experiment.

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS. arXiv:1706.03762.
2. Goodfellow, I., et al. (2014). Generative Adversarial Nets. NeurIPS. arXiv:1406.2661.
3. Kingma, D. P., and Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.
4. Ho, J., Jain, A., and Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS. arXiv:2006.11239.
5. Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI.
6. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.
7. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. NeurIPS. arXiv:2005.14165.
8. Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.
9. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556.
10. Wei, J., et al. (2022). Emergent Abilities of Large Language Models. TMLR. arXiv:2206.07682.
11. Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. NeurIPS. arXiv:2203.02155.
12. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS. arXiv:2005.11401.
13. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

# Result

A comparative evaluation of ChatGPT and Google Gemini was prepared using five prompts covering the fundamentals of Generative AI, its architectures and applications, scaling in Large Language Models, and the process of building LLMs. The evaluation framework considers accuracy, creativity, hallucination, reasoning, speed, and engineering usefulness. The experiment demonstrates the importance of comparing AI-generated responses systematically and verifying technical information before using it for engineering purposes.

---

If the tables paste into Word as tab-separated text, select that text and use Insert > Table > Convert Text to Table. The references are from memory, so check them against the original papers.
