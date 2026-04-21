# Cesar School
# Ciência da Computação

**Théo de Albuquerque Moura**  
**Orientador: Daniel Bezerra**

# Modelagem de Padrões de Opiniões Políticas a partir de Pautas Ideológicas
## Versão expandida a partir do projeto original

## Resumo

Este trabalho investiga se variáveis demográficas ajudam a explicar posições políticas por pauta e, em seguida, em que medida essas pautas ajudam a explicar a escolha de voto na eleição presidencial dos Estados Unidos em 2024. O estudo utiliza microdados do American National Election Studies (ANES) 2024 e organiza a análise em duas etapas: na primeira, busca prever opiniões sobre temas como economia, imigração, armas, clima, aborto, igualdade racial e papéis de gênero a partir de características demográficas; na segunda, utiliza tanto as pautas observadas quanto as pautas estimadas para modelar a escolha entre Harris e Trump. Foram comparados modelos de regressão logística com Elastic Net, Random Forest e XGBoost com validação cruzada estratificada e uso de pesos amostrais. Os resultados mostram que a camada de pautas adiciona informação relevante em relação a modelos puramente demográficos, mas também revelam que identificação partidária continua sendo um dos sinais mais fortes para explicar o voto. As pautas observadas melhoram substancialmente o desempenho em relação à demografia estrita, enquanto as pautas previstas a partir da demografia ajudam apenas parcialmente, indicando que a mediação demografia -> pautas -> voto é plausível, porém imperfeita. Em paralelo, sinais de estilo de vida, mídia e engajamento político aparecem como mais informativos do que variáveis materiais simples de condição de vida. O trabalho contribui ao propor uma leitura interpretável do voto mediado por pautas, mas reconhece limites importantes: a análise é retrospectiva, baseada em atitudes pós-eleição, e não deve ser lida como previsão eleitoral ex ante.

## 1. Introdução

As eleições presidenciais dos Estados Unidos em 2024 reforçaram um problema conhecido na literatura de comportamento eleitoral: modelos baseados apenas em perfil demográfico e identificação agregada de grupos sociais nem sempre explicam de forma satisfatória como o voto é organizado no plano individual. Idade, escolaridade, renda, raça, religião e gênero continuam relevantes, mas parecem insuficientes para capturar a forma pela qual os eleitores articulam suas preferências em torno de temas específicos, como inflação, imigração, aborto, armas, clima, raça e papéis de gênero.

Este trabalho parte da hipótese de que uma parte importante do vínculo entre perfil social e voto é mediada por pautas. Em vez de assumir que demografia leva diretamente ao voto, a proposta é tratar atitudes por tema como um mecanismo intermediário: características demográficas influenciam a probabilidade de um eleitor adotar certas posições políticas, e essas posições, por sua vez, ajudam a estruturar a decisão final de voto. A questão central é, portanto, dupla: até que ponto a demografia explica padrões de opinião por pauta, e até que ponto essas pautas explicam melhor o voto do que a demografia isoladamente.

O repositório deste TCC já contém um pipeline empírico relativamente completo com dados do ANES 2024, preparação de base, experimentos de modelagem, testes de significância, estudos de ablação e medidas de importância de variáveis. Por isso, esta versão expandida não é apenas um projeto de pesquisa: ela reconstrói o trabalho como um estudo empírico efetivamente executado, preservando a motivação teórica do texto original, mas alinhando a redação ao que foi implementado.

## 2. Fundamentação teórica

### 2.1 Demografia, issues e voto

A literatura clássica de comportamento eleitoral oferece pelo menos três pontos de partida para esta pesquisa. O primeiro é a tradição da identificação partidária, em que vínculos políticos relativamente estáveis organizam a percepção dos eleitores sobre candidatos, partidos e acontecimentos. O segundo é a tradição da escolha racional e do voto retrospectivo, em que eleitores avaliam desempenho, proximidade programática ou utilidade esperada ao escolher entre alternativas. O terceiro é a literatura de issue voting, que destaca o papel de temas concretos na organização da competição política.

O argumento deste TCC se aproxima mais do terceiro eixo sem abandonar os dois anteriores. A ideia é que o voto não é produzido apenas por pertencimento social bruto nem exclusivamente por lealdade partidária abstrata; ele também depende de como o eleitor se posiciona diante de questões políticas específicas e da importância subjetiva que essas questões assumem. Em contextos de polarização, mídia fragmentada e forte carga simbólica de temas morais e identitários, essas pautas podem funcionar como pontos de coordenação entre grupos sociais muito distintos.

### 2.2 Saliência e propriedade de pauta

O estudo também dialoga com as noções de saliência e issue ownership. A literatura sugere que dois eleitores podem concordar sobre uma pauta e ainda assim votar de forma distinta se ela tiver pesos diferentes em sua hierarquia de prioridades. Além disso, certos partidos tendem a ser percebidos como mais competentes em temas específicos, o que altera o retorno eleitoral da mesma posição substantiva.

Embora o pipeline implementado no repositório não modele saliência de forma tão rica quanto o projeto original idealizava, a estrutura geral continua compatível com essa literatura: o trabalho parte da ideia de que posições por pauta condensam mecanismos intermediários importantes entre demografia e voto, e que sua presença no modelo melhora a explicação substantiva da escolha eleitoral.

### 2.3 Escopo metodológico real do TCC

O projeto original mencionava uma variedade ampla de métodos, incluindo GAMs, BART, modelos bayesianos hierárquicos e classes latentes. A implementação efetiva do TCC foi mais enxuta, o que é adequado ao escopo de uma monografia de graduação. Na prática, o estudo compara três famílias de modelos em dados tabulares: regressão logística penalizada com Elastic Net, Random Forest e XGBoost. Essa redução de escopo é metodologicamente defensável, porque permite contrastar um modelo linear interpretável com dois modelos não lineares fortes sem dispersar a análise em múltiplas técnicas pouco exploradas.

## 3. Metodologia

### 3.1 Fonte de dados

A base empírica do trabalho é o ANES 2024 Time Series. O arquivo bruto utilizado no repositório contém 5.521 observações e 1.722 colunas. A partir desse material foi construída uma base analítica reduzida, com 93 variáveis selecionadas manualmente, cobrindo:

1. Demografia e posicionamento geral.
2. Opiniões por pauta.
3. Sinais de mídia, estilo de vida e engajamento.
4. Identificação partidária.
5. Pesos amostrais.
6. Variável de voto.

O script `prepare_anes.py` renomeia as variáveis do ANES para nomes interpretáveis, trata códigos negativos como valores ausentes e gera dois conjuntos derivados:

1. `anes_2024_clean.csv`, com 3.693 eleitores e foco nos votantes.
2. `anes_2024_with_abstained.csv`, com 4.971 observações, incorporando abstencionistas.

Apesar da existência da base com abstenção, o pipeline principal de modelagem do voto trabalha apenas com os casos Harris e Trump, removendo a classe "Other" por baixa frequência e não usando a classe "Abstained" no modelo final principal.

### 3.2 Variáveis utilizadas

As variáveis demográficas principais incluem faixa etária, sexo, raça, escolaridade, renda, estado civil, emprego, classe social, frequência religiosa, hispanicidade e uma variável de auto-posicionamento ideológico. As pautas cobrem economia, imigração, armas, clima, aborto, raça, populismo, gênero, comércio, diversidade e saúde. Em paralelo, o repositório também organiza um bloco de estilo de vida e engajamento com consumo de mídia, hábitos televisivos, sites acessados, atenção política e emoções em relação à situação do país.

No pipeline efetivamente executado, 30 variáveis de pauta entraram nos resultados do estágio 1. O conjunto teórico definido em código tinha 31, mas uma delas não gerou saída final no arquivo salvo de resultados.

### 3.3 Estratégia empírica em duas etapas

O desenho analítico tem duas camadas.

**Estágio 1: Demografia -> pautas**  
Para cada pauta, o modelo tenta prever a posição do eleitor a partir das variáveis demográficas. Questões com mais de três categorias são discretizadas em tercis para simplificar a classificação. Cada pauta é modelada separadamente, e o melhor modelo por tema é escolhido com base em acurácia balanceada em validação cruzada.

**Estágio 2: Demografia + pautas -> voto**  
Na etapa seguinte, o alvo é a escolha presidencial entre Harris e Trump. O pipeline compara várias abordagens:

1. Pautas observadas + demografia.
2. Pautas previstas no estágio 1 + demografia, com validação aninhada.
3. Demografia com ideologia.
4. Demografia estrita sem ideologia.
5. Identificação partidária isolada.
6. Identificação partidária + demografia.
7. Pautas isoladas.
8. Modelo completo com partido + pautas + demografia.

Essa comparação é importante porque separa perguntas diferentes. Uma coisa é saber se pautas melhoram a explicação do voto quando observadas diretamente. Outra é saber se a demografia consegue reconstruir essas pautas com qualidade suficiente para preservar o ganho explicativo na etapa seguinte.

### 3.4 Modelos e validação

Os modelos comparados em todas as etapas são:

1. Regressão Logística com penalização Elastic Net.
2. Random Forest.
3. XGBoost.

O protocolo de avaliação usa validação cruzada estratificada em 5 folds, imputação de faltantes por mediana dentro de cada fold, padronização, pesos amostrais pós-eleição e métricas de desempenho como acurácia balanceada, macro-F1, log-loss e Brier score quando aplicável. O uso de predições out-of-fold e a implementação aninhada para a abordagem com pautas previstas são pontos fortes do desenho, porque evitam parte importante do vazamento de informação entre estágios.

Além dos resultados principais, o repositório contém:

1. Testes de significância por bootstrap e McNemar entre abordagens do estágio 2.
2. Estudo de ablação leave-one-out por pauta.
3. Adição cumulativa de pautas com base em importância.
4. Análise de importância via SHAP.
5. Comparação entre sinais de estilo de vida e condições materiais.

## 4. Resultados

### 4.1 O quanto a demografia explica as pautas

Os resultados do estágio 1 mostram um quadro heterogêneo. Em média, a Random Forest foi o melhor método para prever pautas a partir da demografia, com acurácia balanceada média de 0,555. O desvio padrão entre temas foi alto, o que indica que algumas atitudes são muito mais previsíveis do que outras.

As pautas mais previsíveis foram:

1. `climate_impact` com acurácia balanceada de 0,774.
2. `immig_policy3` com 0,722.
3. `climate_regulation` com 0,705.
4. `gun_assault_ban` com 0,705.
5. `min_wage` com 0,689.

As menos previsíveis foram temas como `populism_insiders` (0,377), `trade_summary` (0,381), `econ_mobility` (0,397) e `populism_rich` (0,428). Em outras palavras, algumas opiniões ligadas a clima, armas e certos eixos culturais parecem estar mais fortemente ancoradas em perfis demográficos do que atitudes ligadas a populismo econômico, comércio ou mobilidade percebida.

Esse resultado já é substantivo: ele sugere que a mediação demografia -> pautas não é homogênea. Algumas pautas são facilmente reconstruídas a partir do perfil social; outras parecem depender mais de trajetórias, contextos locais, identidade partidária, informação ou experiências específicas não observadas no conjunto de dados.

### 4.2 O quanto as pautas explicam o voto

A pergunta principal do trabalho aparece no estágio 2. Considerando o melhor modelo de cada abordagem, os resultados foram os seguintes:

| Abordagem | Melhor modelo | Acurácia balanceada |
| --- | --- | ---: |
| Partido + pautas + demografia | Random Forest | 0,951 |
| Partido + demografia | Random Forest | 0,937 |
| Partido apenas | Regressão Logística | 0,934 |
| Pautas observadas + demografia | XGBoost | 0,928 |
| Pautas previstas + demografia | Random Forest | 0,899 |
| Demografia + ideologia | XGBoost | 0,895 |
| Pautas apenas | Random Forest | 0,889 |
| Demografia estrita | Random Forest | 0,876 |

O primeiro achado importante é que pautas observadas melhoram de forma clara a explicação do voto em relação à demografia estrita. A comparação entre `A_actual_issues` e `Cs_demo_strict` mostra diferença média de 0,0638 em acurácia balanceada, com significância estatística forte. Mesmo contra a linha de base com ideologia (`C_demo_with_ideology`), o ganho permanece substantivo e significativo: 0,0333.

O segundo achado é que pautas previstas a partir da demografia preservam apenas parte desse ganho. A abordagem `B_predicted_issues` supera a demografia estrita, mas não fica claramente acima do modelo demográfico com ideologia. Em outras palavras, a ideia de mediação é plausível, porém a primeira etapa ainda não reconstrói as pautas com fidelidade suficiente para reproduzir toda a força explicativa das atitudes observadas.

O terceiro achado é que identificação partidária continua extremamente poderosa. `D_party_only` já alcança desempenho comparável ou superior ao modelo com pautas observadas + demografia, e `G_full_with_party` é o melhor sistema do repositório. Isso é compatível com a literatura de polarização partidária: mesmo quando pautas importam, elas convivem com um alinhamento partidário muito forte.

### 4.3 Testes de significância e leitura substantiva

Os testes de significância reforçam alguns pontos centrais:

1. Pautas observadas + demografia superam pautas previstas + demografia.
2. Pautas observadas + demografia superam os dois baselines puramente demográficos.
3. Pautas observadas + demografia não superam de forma estatisticamente robusta os modelos com partido apenas ou partido + demografia.
4. O modelo completo com partido + pautas + demografia supera o modelo com pautas observadas + demografia.

Substantivamente, isso sugere uma interpretação equilibrada. Não faz sentido dizer que demografia sozinha explica o voto. Também não faz sentido dizer que pautas eliminam a importância de partido. O quadro mais plausível é um eleitorado em que identidades partidárias continuam centrais, mas no qual atitudes temáticas adicionam informação real e ajudam a revelar os mecanismos pelos quais grupos sociais se aproximam dos partidos.

### 4.4 Importância de variáveis e estudo de ablação

A análise de importância global mostra que as variáveis com maior peso no modelo com pautas observadas incluem:

1. `age_group`
2. `sex`
3. `ideology`
4. `climate_impact`
5. `gun_difficulty`
6. `racial_tryharder`
7. `education`
8. `gun_assault_ban`
9. `climate_regulation`
10. `racial_slavery`

Essa combinação é interessante porque mistura demografia, auto-posicionamento ideológico e pautas substantivas. O resultado sugere que o voto de 2024, no recorte estudado, foi especialmente estruturado por idade, sexo, ideologia geral e por eixos ligados a clima, armas e atitudes raciais.

O estudo de ablação leave-one-out indica que as pautas cuja remoção mais prejudica o desempenho são:

1. `abortion_detail`
2. `tax_millionaires`
3. `climate_regulation`
4. `racial_deserve`
5. `diversity_summary`

Nenhuma pauta isolada derruba o modelo drasticamente, o que indica redundância parcial e efeito conjunto. Ainda assim, o padrão geral confirma que temas morais, redistributivos e identitários têm peso não trivial.

Na adição cumulativa, apenas cinco pautas já levam o modelo a cerca de 0,921 de acurácia balanceada, muito próximo do desempenho final com o conjunto completo. Isso sugere que uma parte relativamente pequena do espaço temático concentra grande parcela do sinal preditivo.

### 4.5 Estilo de vida, mídia e condições materiais

Uma extensão importante do repositório compara dois blocos explicativos: sinais de estilo de vida e mídia, de um lado, e condições materiais simples, de outro. Os resultados são expressivos:

1. `all_media_lifestyle`: 0,804 de acurácia balanceada.
2. `conditions_only`: 0,620.
3. `engagement_attitudes_only`: 0,747.
4. `media_conditions_combined`: 0,830.
5. `media_conditions_demo`: 0,890.

O ganho incremental também é assimétrico. Adicionar sinais de estilo de vida ao bloco de condições materiais aumenta a acurácia em cerca de 0,204. Adicionar condições materiais ao bloco de estilo de vida aumenta apenas 0,025. Isso não significa que renda, emprego ou classe deixaram de importar; significa, no entanto, que hábitos de mídia, engajamento político e estados emocionais capturam uma parcela muito maior da variação observada no voto dentro deste conjunto de dados.

Na análise SHAP específica desse bloco, `media_trust`, `emotion_happy`, `econ_opportunity`, `emotion_hope`, `emotion_pride` e programas politizados de TV aparecem entre os sinais mais influentes. O padrão é consistente com a ideia de que o voto contemporâneo é fortemente mediado por consumo de informação, engajamento e climas afetivos, e não apenas por posição econômica objetiva.

## 5. Discussão

### 5.1 Contribuição principal

O principal mérito deste trabalho é mostrar que o modelo demografia -> pautas -> voto faz sentido como estrutura explicativa, ainda que de forma incompleta. Quando as pautas são observadas diretamente, elas melhoram de forma robusta a explicação do voto em relação aos baselines demográficos. Isso valida a intuição teórica central do projeto: a relação entre perfil social e escolha eleitoral é parcialmente mediada por posicionamentos em temas concretos.

Ao mesmo tempo, o estudo evita um exagero frequente. Não basta inserir pautas no modelo para concluir que elas são tudo. A força da variável de partido mostra que o eleitorado estudado continua profundamente ordenado por alinhamento partidário. Na prática, o melhor enquadramento não é "demografia versus pautas versus partido", mas sim a combinação dessas camadas.

### 5.2 O que o estágio 1 revela sobre previsibilidade política

O estágio 1 é importante porque mede algo diferente da etapa final. Ele não pergunta apenas quem vota em quem, mas quão reconstruíveis são certas opiniões políticas a partir do perfil demográfico. O fato de temas como clima e armas serem mais previsíveis do que populismo ou comércio sugere que algumas clivagens estão mais sedimentadas socialmente do que outras. Isso tem valor substantivo próprio: nem toda pauta relevante eleitoralmente é igualmente ancorada em demografia observável.

### 5.3 O papel do estilo de vida e do ambiente informacional

Talvez o resultado mais original do repositório seja a força dos blocos de mídia, engajamento e emoção. Mesmo sem serem o foco central do projeto original, esses sinais parecem capturar melhor a escolha presidencial do que um pequeno conjunto de condições materiais. Isso reforça a ideia de que a política contemporânea é mediada por ecossistemas de informação, repertórios culturais e disposições afetivas, elementos que não cabem facilmente em leituras clássicas centradas apenas em classe, renda ou ocupação.

## 6. Limitações

Este trabalho tem limitações importantes e elas precisam ser explicitadas com clareza.

### 6.1 Retrodicção, não previsão eleitoral ex ante

Grande parte das pautas usadas no pipeline vem de variáveis pós-eleição do ANES 2024. Isso significa que o estudo não deve ser apresentado como um modelo de previsão pré-eleitoral. Ele é, acima de tudo, uma análise retrospectiva da estrutura do voto já ocorrido.

### 6.2 Escopo reduzido da variável de voto

O modelo final principal trabalha com Harris versus Trump e exclui a classe "Other". Além disso, embora exista uma base com abstencionistas, a modelagem principal não trata abstenção como alvo final. O trabalho, portanto, explica um recorte importante da escolha eleitoral, mas não todo o comportamento político possível no survey.

### 6.3 Distância entre projeto ideal e implementação real

O texto original do projeto propunha métodos e operações não implementados no repositório atual, como GAMs, BART, classes latentes, curvas completas de calibração e medidas mais ricas de saliência. A versão expandida deste TCC precisa reconhecer essa diferença e assumir um escopo metodológico mais enxuto, porém real.

### 6.4 Limites inferenciais

Há também restrições metodológicas adicionais:

1. A discretização de algumas pautas em tercis usa um procedimento aproximado.
2. Não existe um holdout externo final por onda, ano ou região.
3. O desenho mede associação preditiva, não causalidade identificada.
4. Parte da explicação do voto pode refletir racionalização pós-eleitoral das respostas por pauta.

Esses limites não invalidam o estudo, mas devem orientar a forma como os resultados são apresentados.

## 7. Conclusão

Este trabalho investigou se a demografia ajuda a explicar atitudes políticas por pauta e se essas pautas ajudam a explicar o voto presidencial nos Estados Unidos em 2024. Os resultados permitem quatro conclusões centrais.

1. Modelos puramente demográficos são insuficientes para capturar toda a estrutura do voto.
2. Pautas observadas acrescentam informação relevante e melhoram significativamente a explicação da escolha eleitoral.
3. Pautas previstas a partir da demografia funcionam apenas parcialmente, o que indica que a mediação proposta é real, mas ainda mal capturada no estágio 1.
4. Identificação partidária permanece central e, em muitos cenários, é tão ou mais forte do que as pautas observadas.

Como contribuição acadêmica, o TCC mostra que uma abordagem em duas etapas é útil para decompor a relação entre perfil social e voto e para distinguir desempenho puramente preditivo de interpretação substantiva. Como contribuição prática, o estudo sugere que análises eleitorais mais informativas precisam combinar demografia, atitudes temáticas e sinais do ambiente informacional, em vez de confiar em apenas uma dessas camadas.

## 8. Encaminhamentos para a versão final

Para transformar esta versão expandida em uma monografia final coesa, os próximos passos mais importantes são:

1. Integrar este texto ao padrão formal exigido pela instituição.
2. Inserir tabelas e figuras derivadas dos arquivos em `results/`.
3. Reduzir a fundamentação teórica ao que realmente sustenta a análise executada.
4. Escrever uma seção curta e honesta sobre por que o escopo implementado foi reduzido em relação ao projeto original.
5. Se houver tempo, rodar uma extensão simples com abstenção ou com separação entre perguntas pré e pós-eleição.

## Referências essenciais

American National Election Studies (ANES). Time Series Study 2024 Dataset.

Breiman, L. Random Forests. *Machine Learning*, 2001.

Campbell, A. et al. *The American Voter*. 1960.

Downs, A. *An Economic Theory of Democracy*. 1957.

Fiorina, M. *Retrospective Voting in American National Elections*. 1981.

Lundberg, S. M.; Lee, S. I. A Unified Approach to Interpreting Model Predictions. 2017.

Zou, H.; Hastie, T. Regularization and Variable Selection via the Elastic Net. 2005.
