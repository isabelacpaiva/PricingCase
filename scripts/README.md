# Projeto de Pricing de Seguros

Este projeto utiliza a base `data_synthetic.csv`, disponível em [Kaggle](https://www.kaggle.com/datasets/ravalsmit/insurance-claims-and-policy-data), para construir um estudo de precificação de seguros, aplicando técnicas de ciência de dados e fundamentos atuariais. O objetivo é demonstrar habilidades práticas na manipulação de dados, análise de risco, modelagem de sinistralidade e visualização de indicadores.

A base de dados contém informações detalhadas sobre clientes e apólices, incluindo idade, gênero, estado civil, ocupação, nível de renda, educação, localização, histórico de sinistros, tipo de apólice, valor de cobertura, prêmio, dedutível, perfil de risco e score de crédito. A riqueza dessas variáveis permite simular um processo completo de análise de risco e precificação.

O fluxo do projeto é estruturado da seguinte forma:

# 1. Preparação e limpeza dos dados
   - Seleção de variáveis: mantidas apenas as que são úteis para análise de risco e precificação.
    - Criação de `has_claim`: variável binária que indica ocorrência de sinistro, fundamental para modelagem de frequência.
    - Criação de `claim_frequency` e `claim_severity`: separação entre  frequência e severidade, abordagem clássica em pricing.
    - Criação de `loss_ratio`: indicador técnico de sinistralidade, essencial para avaliar se o prêmio cobre o risco.
    - Tratamento de nulos: categorias → "Unknown"; numéricas → mediana (robusto contra outliers).
   
- **Criação de variáveis derivadas**  
   - `has_claim`: indicador binário de ocorrência de sinistro.  
   - `claim_frequency`: frequência de sinistros anteriores do cliente.  
   - `claim_severity`: severidade média dos sinistros.  
   - `loss_ratio`: sinistralidade, calculada como sinistro dividido pelo prêmio.  

# 2. Análise Exploratória (EDA)

A análise exploratória de dados foi realizada para compreender a estrutura da base, identificar padrões de risco e relacionar variáveis-chave para pricing de seguros. Cada passo da EDA buscou gerar insights acionáveis para modelagem futura e visualização de indicadores.

- **Distribuição de variáveis demográficas e de risco**
  - **Objetivo:** Conhecer a composição da base, entender perfis de clientes e detectar possíveis enviesamentos ou agrupamentos.  
  - **O que foi feito:** Foram analisadas distribuições de idade, score de crédito, prêmio, cobertura, ocupação, estado civil e perfil de risco.  
  - **Insights obtidos:**  
    - A base apresenta diversidade de ocupações, perfis de risco e estado civil, o que permite segmentações diferenciadas.  
    - Variáveis contínuas como idade, score de crédito e prêmio apresentam correlação quase nula com a ocorrência de sinistro (`has_claim`), sugerindo que fatores históricos de sinistros e perfil de risco são mais determinantes para o risco.

- **Relações entre frequência, severidade e perda relativa ao prêmio**
  - **Objetivo:** Separar o impacto da **frequência de sinistros** (quantidade de eventos) da **severidade** (valor financeiro dos sinistros) e da **sinistralidade relativa ao prêmio** (loss ratio), seguindo a lógica clássica atuária de pricing.  
  - **O que foi feito:** Foram calculadas e analisadas as variáveis `has_claim`, `claim_frequency`, `claim_severity` e `loss_ratio`, além de explorar a relação com o `premium_amount`.  
  - **Insights obtidos:**  
    - Clientes com maior severidade de sinistro (`claim_severity`) e maior sinistralidade (`loss_ratio`) apresentam maior probabilidade de ter sinistro.  
    - Maior prêmio (`premium_amount`) tende a estar associado a menor `loss_ratio`, indicando que apólices de maior valor possuem risco relativo menor.  
    - Frequência e severidade apresentam correlação negativa, evidenciando que clientes com muitos sinistros tendem a ter eventos de menor impacto, enquanto clientes com poucos sinistros podem ter eventos de alto impacto.  
    - Esses padrões reforçam a necessidade de separar **frequência** e **severidade** no modelo de precificação para capturar corretamente o risco esperado.

- **Identificação de padrões de risco por segmentos de clientes**
  - **Objetivo:** Avaliar como variáveis categóricas como estado civil, ocupação e perfil de risco se relacionam com a probabilidade de ocorrência de sinistro, permitindo segmentações e ajustes de prêmio diferenciados.  
  - **O que foi feito:** Foram calculadas taxas de sinistralidade (`has_claim`) por `marital_status`, `occupation` e `risk_profile`.  
  - **Insights obtidos:**  
    - Determinados segmentos apresentam taxas de sinistralidade significativamente maiores. Por exemplo, perfis de risco mais altos e algumas ocupações específicas mostraram maior propensão a sinistros.  
    - Essas diferenças segmentadas podem orientar estratégias de pricing diferenciadas, permitindo alocar prêmios mais adequados ao risco de cada grupo de clientes.  

- **Correlação entre variáveis numéricas**
  - **Objetivo:** Identificar variáveis mais relacionadas com frequência, severidade e sinistralidade, servindo de base para seleção de features no modelo de pricing.  
  - **O que foi feito:** Calculou-se a correlação entre variáveis numéricas (`age`, `credit_score`, `premium_amount`, `coverage_amount`, `deductible`, `claim_frequency`, `claim_severity`, `loss_ratio`, `has_claim`).  
  - **Insights obtidos:**  
    - `claim_severity` e `loss_ratio` possuem correlação positiva moderada com `has_claim`, reforçando seu papel como indicadores de risco.  
    - `premium_amount` apresenta correlação negativa com `loss_ratio`, indicando que clientes com apólices de maior prêmio têm menor sinistralidade relativa.  
    - Outras variáveis como idade, score de crédito, cobertura e deductible apresentam correlação quase nula com sinistralidade e ocorrência de sinistro, sugerindo que isoladamente elas têm menor poder explicativo, mas podem contribuir em modelos multivariados ou segmentações.  
    - A correlação negativa entre `claim_frequency` e `claim_severity` reforça o padrão observado de frequência × severidade típico em dados atuariais.

# Modelagem de Severidade de Sinistros com GLM Tweedie e Redes Neurais

A modelagem de custos de sinistros em seguros não-vida apresenta desafios particulares devido à natureza dos dados, que frequentemente apresentam concentração de valores nulos, forte assimetria à direita e caudas pesadas (Ohlsson & Johansson, 2010). Nesse contexto, utilizamos duas abordagens complementares: modelos lineares generalizados (GLMs) com distribuição Tweedie e redes neurais artificiais.

## Fundamentação Teórica

A distribuição Tweedie pertence à família exponencial e é apropriada para variáveis contínuas não negativas com massa em zero. Sua função de variância é dada por:

$$
Var(Y) = \phi \mu^p, \quad 1 < p < 2
$$

onde $\mu = E(Y)$, $\phi$ é o parâmetro de dispersão e $p$ controla a relação entre média e variância. Valores de $p$ entre 1 e 2 permitem modelar a mistura Poisson-Gama, que é apropriada para dados de seguros de ramos elementares (Jørgensen, 1997).

O GLM assume a forma:

$$
g(\mu_i) = \mathbf{x}_i^\top \beta
$$

onde $g(\cdot)$ é a função de ligação (logarítmica neste caso), $\mathbf{x}_i$ representa o vetor de covariáveis e $\beta$ o vetor de parâmetros a serem estimados. A estimação é realizada via máxima verossimilhança.

Por outro lado, as redes neurais artificiais constituem modelos não paramétricos capazes de capturar relações não lineares complexas entre covariáveis e a variável resposta. Consideramos aqui uma arquitetura *feedforward* do tipo *Multilayer Perceptron* (MLP), com camadas densas, funções de ativação não lineares (ReLU) e regularização via *dropout*. O treinamento foi realizado com *backpropagation* utilizando o otimizador Adam.

## Diagnóstico de Multicolinearidade e Zero-Inflation

A multicolinearidade pode ser avaliada pelo *Variance Inflation Factor* (VIF):

$$
VIF_j = \frac{1}{1 - R_j^2}
$$

onde $R_j^2$ é o coeficiente de determinação da regressão da variável $j$ sobre todas as demais. Valores acima de 10 sugerem multicolinearidade severa (Kutner et al., 2004).

A presença de excesso de zeros (zero-inflation) pode ser investigada comparando-se a proporção de observações nulas com a prevista pelo modelo Tweedie. Caso haja discrepância, alternativas incluem modelos *hurdle* ou *zero-inflated Tweedie* (Zhang, 2013).

## Métricas Específicas em Seguros

Além de métricas clássicas como *Root Mean Squared Error* (RMSE), *Mean Absolute Error* (MAE) e $R^2$, em seguros é relevante avaliar medidas como:

- **Deviance residuals**: indicam o ajuste do GLM Tweedie.
- **Gini index** e **Lift curves**: úteis para avaliar capacidade discriminatória entre riscos altos e baixos.
- **Loss ratio** por grupos de risco: para verificar viés sistemático no ajuste.

Essas métricas permitem não apenas avaliar a acurácia preditiva, mas também a utilidade prática do modelo na tarifação.

## Resultados Obtidos

O GLM Tweedie ajustado apresentou estimativas consistentes com a literatura, capturando adequadamente a distribuição de massa em zero e a assimetria dos custos de sinistro. Contudo, análises preliminares de resíduos indicaram possível presença de multicolinearidade entre algumas covariáveis, sugerindo necessidade de avaliação via VIF e possível regularização ou seleção de variáveis.

A rede neural apresentou desempenho preditivo muito elevado, com valores de RMSE e MAE bastante reduzidos e $R^2$ próximo de 1, tanto em treino quanto em teste. Essa performance, ainda que positiva, pode indicar *overfitting*, especialmente dado o forte desbalanceamento da variável resposta. O uso de *early stopping* e *dropout* contribuiu para mitigar esse risco, mas o comportamento da função de perda deve ser monitorado atentamente. 

Além disso, a interpretação de redes neurais é menos direta que em GLMs. Para viabilizar sua aplicação prática em seguros, técnicas de interpretação como *permutation importance* ou *SHAP values* podem ser aplicadas, possibilitando entender a contribuição relativa das covariáveis.

Em termos comparativos, enquanto o GLM Tweedie oferece interpretabilidade e fundamentação teórica clássica para tarifação, a rede neural provê maior poder preditivo, embora com menor transparência. Uma estratégia recomendada seria considerar o GLM como *benchmark* e a rede neural como modelo de ganho, aplicando validações adicionais para garantir robustez.

## Referências

- Jørgensen, B. (1997). *The Theory of Dispersion Models*. Chapman & Hall.
- Ohlsson, E., & Johansson, B. (2010). *Non-Life Insurance Pricing with Generalized Linear Models*. Springer.
- Zhang, Y. (2013). Likelihood-based and Bayesian methods for Tweedie compound Poisson linear mixed models. *Statistics and Computing*, 23(6), 743–757.
- Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2004). *Applied Linear Statistical Models*. McGraw-Hill Irwin.

