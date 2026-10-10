# Tech Challenge — Fase 2 | POSTECH Data Analytics

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | 2DTATBB |
| Grupo | Grupo 43 |
| Data de entrega | 10/10/2026|

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| Rejane Brito de Carvalho  | RM377775 | rejanebrito@bb.com.br |
| ERICA DE ABREU ROCHA BORGES| RM377819 | erica_rocha@bb.com.br |
| JOSIAS MATOS DE ARAUJO | RM377801 | matosdearaujo@bb.com.br|
| MONIQUE OLIVEIRA PEREIRA | RM377781 | mpesousa@yahoo.com |
| DEBORA TEIXEIRA PINTO | RM377832 | debora.tex@bb.com.br |

---

## 2. Links da entrega

Estes três links são **obrigatórios** e devem ser idênticos aos do PDF de submissão.

| Item | Link |
|---|---|
| Repositório | <!-- PREENCHER: URL pública do GitHub --> |
| Vídeo executivo (≤ 5 min) | https://youtu.be/4lYs1__UmZ0 |
| Apresentação | `docs/apresentacao_executiva.pdf` ou https://canva.link/qlxxlkhp8xp6k5q |

> ⚠️ Repositório privado ou inacessível **zera** toda a Dimensão 1 da rúbrica.
> Confira o acesso em uma janela anônima antes de enviar.

---

## 3. O problema

A instituição precisa decidir quais solicitações de crédito aprovar ou recusar, sabendo que os erros têm custos assimétricos: aprovar um mau pagador gera perda do principal (calote), enquanto recusar um bom pagador gera perda de receita. Com uma base extremamente desbalanceada — 98,3% de bons pagadores contra apenas 1,7% de maus — decisões manuais ou baseadas em regras fixas são ineficientes e não escalam.

A motivação para usar Machine Learning está na natureza comportamental e multivariada do risco. A EDA mostrou que o comportamento passado de pagamento (ex.: `pct_quitado`, `n_meses`) é o preditor mais forte de inadimplência, superando variáveis cadastrais estáticas como renda e idade. Modelos de ML capturam interações não-lineares complexas entre dezenas de *features* que regras manuais não conseguem mapear, além de exigirem técnicas específicas para o desbalanceamento extremo (SMOTE, `scale_pos_weight`) e métricas adequadas (Recall, F1, AUC-ROC).

O objetivo não é substituir o analista, mas construir uma peneira inteligente que organize a fila de decisão, priorize casos de alto risco para análise humana e torne o processo mensurável, auditável e alinhado ao apetite de risco da instituição.

Em resumo: o problema é transformar um histórico de crédito ruidoso e desbalanceado em uma ferramenta preditiva que minimize o custo total do risco — algo inviável sem a capacidade de generalização do Machine Learning.

---

### Variável alvo

**Definição:** `eh_mau_pagador` — variável binária em que:
* `0` = Bom Pagador
* `1` = Mau Pagador

**Construção:** a partir da base `credit_record.csv` (`ID`, `MONTHS_BALANCE`, `STATUS`). Cada cliente possui múltiplos registros mensais de crédito. A regra de rotulagem aplicada foi:

| STATUS original | Significado | Rótulo binário |
| :--- | :--- | :--- |
| `0`, `1`, `C`, `X` | Atraso leve (até 59 dias), quitado ou sem empréstimo | `0` (Bom Pagador) |
| `2`, `3`, `4`, `5` | Atraso igual ou superior a 60 dias | `1` (Mau Pagador) |

Após a binarização, os registros foram agregados por ID com `groupby("ID")["eh_mau_pagador"].max().reset_index()`, garantindo um rótulo único por cliente.

**Limiar adotado:** 60 dias de atraso ($\text{STATUS} \ge 2$).

**Justificativa do limiar:**
* Atrasos de 1 a 29 dias (STATUS 0) e 30 a 59 dias (STATUS 1) são comuns, geralmente ligados a esquecimento, descasamento de datas ou aperto temporário de fluxo de caixa. Têm alta taxa de regularização voluntária e não configuram perda efetiva para o credor.
* A partir de 60 dias (STATUS 2, 3, 4, 5), a probabilidade de recuperação despenca e a chance de calote definitivo (*write-off*) cresce exponencialmente. Esse é o ponto de inflexão entre um problema de liquidez passageiro e uma insolvência estrutural, alinhado à prática de mercado.

**Justificativa da agregação por `.max()`:** adota-se o princípio conservador de risco. Se o cliente atingiu 60 dias ou mais de atraso em qualquer mês do histórico, ele é marcado como Mau Pagador (`1`). Isso impede que um histórico majoritariamente adimplente mascare um evento grave de inadimplência.

**Distribuição das classes:** após a agregação, a base apresenta 98,3% de Bons Pagadores (`0`) e 1,7% de Maus Pagadores (`1`) — ou, antes do *merge* final, aproximadamente 98,55% vs 1,45%. Trata-se de um desbalanceamento extremo, que implica:
* Uso de métricas adequadas: Recall, F1-Score e AUC-ROC, e não acurácia pura;
* Aplicação de técnicas de balanceamento, como SMOTE e/ou `scale_pos_weight`;
* Cautela na interpretação de acurácia, pois um modelo trivial que prevê sempre “Bom Pagador” atingiria $\approx 98\%$ de acerto, mas seria inútil para o negócio.

---

## Dataset

| Campo | Valor |
| :--- | :--- |
| **Fonte** | Credit Card Approval Prediction — Kaggle |
| **Linhas × colunas** | • `application_record.csv`: 438.557 × 18<br>• `credit_record.csv`: 1.048.575 × 3<br>• **Após merge e tratamento**: **36.457 × 25** |
| **Período / versão** | Histórico mensal de crédito com janela de até 60 meses por cliente (`MONTHS_BALANCE` de −60 a 0, sendo 0 o mês atual) |
| **Licença de uso** | CC0: Public Domain |

### Descrição das Variáveis

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `ID` | `int64` | Identificador único do cliente — chave de junção entre as duas bases |
| `CODE_GENDER` | `object` | Gênero do cliente (M = masculino, F = feminino) |
| `FLAG_OWN_CAR` | `object` | Indica se o cliente possui carro próprio (Y/N) |
| `FLAG_OWN_REALTY` | `object` | Indica se o cliente possui imóvel próprio (Y/N) |
| `CNT_CHILDREN` | `int64` | Quantidade de filhos — removida por redundância com `CNT_FAM_MEMBERS` ($r = 0,88$) |
| `AMT_INCOME_TOTAL` | `float64` | Renda anual declarada pelo cliente (em unidades monetárias) |
| `NAME_INCOME_TYPE` | `object` | Tipo de renda (Working, Pensioner, Commercial associate, State servant, etc.) |
| `NAME_EDUCATION_TYPE` | `object` | Nível de escolaridade (Higher education, Secondary, Incomplete higher, etc.) |
| `NAME_FAMILY_STATUS` | `object` | Estado civil (Married, Single, Civil marriage, Separated, Widow) |
| `NAME_HOUSING_TYPE` | `object` | Tipo de moradia (House / apartment, With parents, Municipal apartment, etc.) |
| `DAYS_BIRTH` | `int64` | Idade em dias (valores negativos, contados a partir da data da aplicação) — convertida para `AGE_YEARS` |
| `DAYS_EMPLOYED` | `int64` | Tempo de emprego em dias (valores negativos). Contém o valor sentinela 365.243 para inativos/aposentados |
| `FLAG_MOBIL` | `int64` | Indica se o cliente possui celular — constante (100% = 1), removida por variância zero |
| `FLAG_WORK_PHONE` | `int64` | Indica se o cliente possui telefone comercial |
| `FLAG_PHONE` | `int64` | Indica se o cliente possui telefone residencial |
| `FLAG_EMAIL` | `int64` | Indica se o cliente possui e-mail cadastrado |
| `OCCUPATION_TYPE` | `object` | Ocupação profissional (~30% de nulos, preenchidos com "Desconhecido") |
| `CNT_FAM_MEMBERS` | `float64` | Número de membros da família |
| `MONTHS_BALANCE` | `int64` | Mês de referência do registro de crédito (−60 a 0; 0 = mês atual) |
| `STATUS` | `object` | Situação mensal do crédito: 0–5 (atrasos crescentes), C (quitado no mês), X (sem empréstimo no mês) |
| `eh_mau_pagador` | `int64` | Variável alvo binária — 1 se o cliente atingiu $\ge 60$ dias de atraso em qualquer mês; 0 caso contrário |
| `IS_EMPLOYED_MISSING` | `int64` | Flag derivada de `DAYS_EMPLOYED` — 1 para clientes inativos/aposentados (valor sentinela 365.243) |
| `AGE_YEARS` | `float64` | Idade em anos (derivada de `DAYS_BIRTH` dividido por −365,25) |
| `n_meses` | `float64` | Total de meses com histórico de crédito registrado (exposição ao crédito) |
| `mes_inicial` | `float64` | Primeiro mês do histórico do cliente (mais distante no passado) |
| `mes_final` | `float64` | Último mês do histórico (geralmente 0 = mês atual) |
| `n_X` | `float64` | Quantidade de meses sem empréstimo ativo (`STATUS = X`) |
| `n_C` | `float64` | Quantidade de meses com crédito quitado (`STATUS = C`) |
| `n_0` | `float64` | Quantidade de meses com atraso leve de 1 a 29 dias (`STATUS = 0`) |
| `pct_quitado` | `float64` | Proporção de meses em que o crédito foi quitado (`n_C / n_meses`) — normaliza o comportamento pelo tempo de exposição |

---

## 4. Como reproduzir

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Baixe o dataset e coloque o arquivo bruto em `data/raw/` (os dados **não** são versionados —
veja `data/README.md`).

Depois execute os notebooks nesta ordem:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória |
| 2 | `notebooks/02_preprocessamento.ipynb` | Limpeza, escala e feature engineering |
| 3 | `notebooks/03_modelagem.ipynb` | Treino e comparação dos modelos |
| 4 | `notebooks/04_avaliacao.ipynb` | Métricas, importância de variáveis e conclusões |

**Semente fixa:** `RANDOM_STATE = 42`, declarada na primeira célula de cada notebook.
Rodar os notebooks na ordem acima, a partir de um ambiente limpo, deve reproduzir
exatamente os números da seção 5.

---

## 5. Resultados

### 5.1 Validação cruzada (5-fold, F1-Score na classe minoritária)

Todos os modelos foram avaliados com `cross_val_score(..., cv=5, scoring="f1")` sobre o conjunto de treino, usando `Pipeline`/`ImbPipeline` para evitar vazamento de dados (*scaler* e *SMOTE* ajustados apenas no fold de treino).

| Modelo | F1-Score (CV) | Desvio padrão | Observação |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | 0,4608 | $\pm$ 0,0253 | Boa performance, estável |
| **KNN** | 0,2517 | $\pm$ 0,0374 | Pior performance, alta variância |
| **SVC** | 0,4704 | $\pm$ 0,0216 | Segundo melhor F1, porém lento |
| **Decision Tree** | 0,1334 | $\pm$ 0,0066 | Muito instável |
| **Random Forest** | 0,1893 | $\pm$ 0,0214 | Indícios de *overfitting* |
| **XGBoost** | **0,5826** | $\pm$ 0,0367 | Melhor F1-Score |

---

### 5.2 Desempenho no conjunto de teste (XGBoost, threshold = 0,30)

| Modelo | Acurácia | Precisão (classe 1) | Recall (classe 1) | F1 (classe 1) | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0,99 | 0,62 | 0,54 | 0,58 | 0,9324 |

#### Leitura por classe:

| Classe | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Bom Pagador (0)** | 0,99 | 0,99 | 0,99 | 7.169 |
| **Mau Pagador (1)** | 0,62 | 0,54 | 0,58 | 123 |
| **Acurácia** | — | — | 0,99 | 7.292 |

---

### 5.3 Modelo escolhido

* **Modelo escolhido:** **XGBoost** (`XGBClassifier` com `scale_pos_weight` ajustado e pipeline com `log1p` em `AMT_INCOME_TOTAL`, `StandardScaler` e `SMOTE`).
* **Por quê:**
  * Maior F1-Score em validação cruzada (`0,5826`) — margem significativa ($> 0,11$) sobre o segundo colocado (`SVC`, `0,4704`) e o terceiro (`Logistic Regression`, `0,4608`).
  * A diferença não é ruído: o desvio padrão entre os *folds* é de apenas `0,0367`, indicando desempenho estável.
  * `AUC-ROC` de `0,93` no teste — excelente capacidade de separação entre as classes, muito acima do limiar de `0,85` considerado "forte" em crédito.
  * Captura relações não-lineares entre *features* comportamentais e cadastrais que modelos lineares (`Logistic Regression`) não conseguem representar.
  * Regularização nativa (`max_depth`, `learning_rate`, `subsample`) e *boosting* combinam árvores fracas em um modelo forte, mitigando o *overfitting* (observado no `Random Forest`).

---

### 5.4 Métricas priorizadas

Em uma base com 98,3% de Bons Pagadores e apenas 1,7% de Maus Pagadores, a acurácia é enganosa: um modelo trivial que sempre prevê "Bom Pagador" atinge ~98% de acerto, mas não identifica nenhum caloteiro. Por isso, priorizaram-se:

* **Recall** — a métrica mais importante:
  * Mede a capacidade de capturar os verdadeiros maus pagadores.
  * O erro de Falso Negativo (aprovar um caloteiro) gera perda direta do principal emprestado — o erro mais caro no negócio.
  * Cada ponto percentual de recall representa menos dinheiro perdido em calotes.
* **Precisão** — controle do falso positivo:
  * Garante que os clientes sinalizados como alto risco realmente sejam de risco.
  * O erro de Falso Positivo (recusar um bom pagador) gera perda de receita (juros cessantes) e insatisfação do cliente.
  * Equilíbrio necessário para não destruir o volume de negócios.
* **F1-Score**:
  * Média harmônica entre *Precision* e *Recall*.
  * Métrica adequada para bases desbalanceadas, pois ignora os verdadeiros negativos (que dominam a base) e foca no desempenho sobre a classe minoritária.
  * Usada como critério principal na validação cruzada.
* **AUC-ROC**:
  * Avalia a capacidade discriminatória do modelo em todos os *thresholds* possíveis.
  * Independente do ponto de corte escolhido, permite comparar modelos de forma justa.
  * *Ressalva:* em desbalanceamento extremo, o AUC pode ser otimista porque o eixo FPR é diluído pela enorme quantidade de verdadeiros negativos. Nesses casos, PR-AUC (*Precision-Recall AUC*) é mais informativa.

#### Por que não usar acurácia pura:
* Um modelo que sempre prevê "Bom Pagador" atinge 98,3% de acurácia e é completamente inútil para o negócio.
* Em risco de crédito, o custo dos erros é assimétrico: o Falso Negativo (calote) custa o principal emprestado, tipicamente muito mais que o lucro cessante de um Falso Positivo.
* O *threshold* de decisão (`0,30`, neste caso) deve ser ajustado pela análise de custo de negócio, não por uma métrica técnica isolada.

---

## 6. Principais Conclusões

1. **O comportamento passado do cliente é o melhor preditor de inadimplência.** As variáveis mais importantes do modelo são `mes_final`, `n_meses` e `pct_quitado` — todas comportamentais. Na prática, isso significa que como o cliente pagou no passado importa mais do que quem ele é: renda, idade e escolaridade tiveram peso quase nulo. Para quem opera a carteira, a prioridade passa a ser coletar e manter histórico de pagamento confiável, e não apenas enriquecer o cadastro com dados demográficos.
2. **O modelo é uma peneira, não um juiz.** Quando aponta um cliente como alto risco, acerta cerca de 6 em cada 10 vezes ($\text{precision} = 0,62$); quando aponta como baixo risco, acerta quase sempre ($99\%$). Mas deixa passar quase metade dos maus pagadores ($\text{recall} = 0,54$). Traduzindo para a operação: o modelo não deve bloquear crédito automaticamente, mas priorizar a fila de análise humana — casos de altíssimo risco seguem para recusa direta, casos ambíguos vão para o analista, e casos de risco muito baixo podem ter aprovação automática.
3. **O ponto de corte é uma decisão de negócio, não técnica.** Aprovar um mau pagador custa o principal emprestado (perda direta); recusar um bom pagador custa o lucro cessante (juros que não serão cobrados). Como o primeiro erro tipicamente custa muito mais caro, o *threshold* deve ser ajustado para baixo — aceitando mais falsos positivos em troca de capturar mais caloteiros. Esse ajuste é uma escolha de política de crédito, alinhada ao apetite de risco da instituição, e não uma decisão do modelo.
4. **O modelo não cobre todos os clientes.** Como depende de *features* comportamentais do histórico de crédito, ele só avalia clientes que já tiveram crédito ativo — hoje, menos de $11\%$ dos cadastrados. Cliente novo, sem histórico, exige regra própria e mais conservadora, ou um segundo modelo baseado em cadastro e bureau externo.
5. **O modelo precisa ser vigiado.** Economia, comportamento e política de crédito mudam. Um modelo treinado hoje pode degradar em seis meses. Monitoramento contínuo, revisão periódica de métricas e re-treino são obrigatórios — não como etapa opcional, mas como parte do ciclo de vida do produto.

---

## 7. Estrutura do repositório

```
.
├── data/          dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/     análise em ordem numerada
└── docs/          apresentação executiva
```

Detalhes e convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviar, percorra o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

### Ambiente e Linguagem

| Tecnologia | Versão | Uso no projeto |
| :--- | :---: | :--- |
| **Python** | 3.11+ | Linguagem base de todos os notebooks |
| **Jupyter Notebook** | — | Ambiente de desenvolvimento e execução dos notebooks |
| **venv** | — | Isolamento de dependências (ambiente virtual) |

---

### Manipulação e Análise de Dados

| Biblioteca | Uso no projeto |
| :--- | :--- |
| **pandas** | Carregamento, limpeza, *merges*, agregações e *feature engineering* |
| **numpy** | Operações vetorizadas, `log1p`, geração de dados sintéticos |

---

### Visualização

| Biblioteca | Uso no projeto |
| :--- | :--- |
| **matplotlib** | Gráficos base (histogramas, *boxplots*, curvas ROC) |
| **seaborn** | *Heatmaps* de correlação, *boxplots* agrupados, estilização (`whitegrid`) |

---

### Machine Learning

| Biblioteca | Uso no projeto |
| :--- | :--- |
| **scikit-learn** | *Pipelines*, `StandardScaler`, `ColumnTransformer`, `train_test_split`, `cross_val_score`, métricas (`classification_report`, `roc_auc_score`, `RocCurveDisplay`, `ConfusionMatrixDisplay`) e modelos lineares/árvores (`Logistic Regression`, `KNN`, `SVC`, `Decision Tree`, `Random Forest`) |
| **imbalanced-learn** | `SMOTE` e `ImbPipeline` — balanceamento da classe minoritária dentro do *pipeline* (sem vazamento de dados) |
| **xgboost** | Modelo vencedor (`XGBClassifier`) com `scale_pos_weight` para lidar com o desbalanceamento extremo |

---

### Serialização e Persistência

| Biblioteca | Uso no projeto |
| :--- | :--- |
| **joblib** | Salvamento e carregamento do melhor modelo (`melhor_modelo.pkl`) |

---

### Bibliotecas Padrão (Python)

| Módulo | Uso no projeto |
| :--- | :--- |
| **os** | Verificação de existência de arquivos e caminhos *fallback* |
| **pathlib** | Manipulação de caminhos (`Path`) — portável entre sistemas |

---

### Estrutura do Projeto

```text
requirements.txt     dependências do projeto (instaladas via pip)
data/raw/            dados brutos (application_record.csv, credit_record.csv)
data/processed/      dataset tratado e modelo serializado
notebooks/           análise em 4 etapas numeradas
docs/                apresentação executiva
