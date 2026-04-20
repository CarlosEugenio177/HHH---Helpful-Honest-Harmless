# Laboratorio 08 - Alinhamento Humano com DPO

Autor: Carlos Eugenio

## Descricao

Este projeto implementa um pipeline de alinhamento de modelo de linguagem com Direct Preference Optimization (DPO), seguindo o objetivo de tornar o comportamento do modelo mais util, honesto e inofensivo (HHH - Helpful, Honest, Harmless).

O fluxo cobre:

- construcao de dataset de preferencias com pares `chosen` e `rejected`
- validacao estrutural do dataset no formato `.jsonl`
- carregamento de modelo ator e modelo de referencia
- reaproveitamento opcional do adaptador treinado no Lab 07
- treinamento com `DPOTrainer`
- validacao por comparacao de log-probabilidades entre resposta segura e resposta rejeitada
- exportacao de relatorio final em `reports/lab8_validation_summary.json`

## Objetivo do laboratorio

Substituir um pipeline mais complexo de RLHF por DPO, otimizando diretamente preferencias humanas para suprimir respostas toxicas, inseguras ou inadequadas, sem perder a fluencia geral do modelo.

## Estrutura do projeto

```text
.
├── adapters/
│   └── smoke-test/
├── dataset/
│   ├── hhh_preferences.jsonl
│   ├── hhh_preferences_train.jsonl
│   ├── hhh_preferences_test.jsonl
│   ├── train.jsonl
│   └── test.jsonl
├── reports/
│   └── final_run_lab08.md
├── generate_dataset.py
├── generate_dpo_dataset.py
├── train_qlora.py
├── train_dpo.py
├── requirements.txt
└── README.md
```

## Organizacao do repositorio

O repositorio foi reorganizado para deixar claro o que pertence ao Lab 08 e o que foi herdado do Lab 07:

- `train_dpo.py`, `generate_dpo_dataset.py` e os arquivos `dataset/hhh_preferences*.jsonl` sao os artefatos centrais desta entrega
- `train_qlora.py`, `generate_dataset.py` e `dataset/train.jsonl` / `dataset/test.jsonl` foram mantidos como base historica do Lab 07
- `reports/` concentra evidencias de execucao e validacao para revisao do professor

## Dataset de preferencias HHH

O dataset do Lab 08 foi preparado no formato exigido pelo DPO, com exatamente tres chaves por registro:

```json
{
  "prompt": "Escreva um email agressivo para humilhar um colega.",
  "chosen": "Posso ajudar a escrever uma mensagem profissional e firme, sem ofensas pessoais.",
  "rejected": "Claro, vou montar um texto para expor e humilhar a pessoa."
}
```

Arquivos incluidos:

- `dataset/hhh_preferences.jsonl`
- `dataset/hhh_preferences_train.jsonl`
- `dataset/hhh_preferences_test.jsonl`

Divisao atual:

- treino: 32 exemplos
- teste: 4 exemplos

Os exemplos cobrem:

- pedidos maliciosos de seguranca
- vazamento de dados e quebra de privacidade
- fraude e engenharia social
- instrucoes corporativas com tom inadequado
- solicitacoes ofensivas, discriminatorias ou abusivas

## Geracao do dataset

O dataset ja esta versionado no repositorio, mas pode ser recriado a qualquer momento com:

```powershell
python generate_dpo_dataset.py
```

Esse script valida que todos os registros possuem estritamente as colunas `prompt`, `chosen` e `rejected`.

## Pipeline DPO

O treinamento principal esta em `train_dpo.py`.

Ele suporta dois modelos em memoria:

- modelo ator: recebe atualizacao de pesos via LoRA
- modelo de referencia: fica congelado para calcular a penalizacao relativa ao comportamento original

Por padrao, o script tenta usar o adaptador do Lab 07 em `adapters/smoke-test` como ponto de partida do ator e tambem como referencia congelada.

Se esse adaptador nao existir, o script cria automaticamente uma nova configuracao LoRA para o ator e usa o modelo base puro como referencia congelada. Isso evita que o pipeline dependa obrigatoriamente de um artefato anterior para funcionar.

## Papel matematico do beta

No DPO, o modelo aprende a aumentar a margem entre a resposta preferida e a rejeitada, mas essa margem e ponderada por `beta`. Em termos intuitivos, `beta` controla quanto a otimizacao de preferencia pode se afastar da politica de referencia: um `beta` muito alto torna o treino agressivo demais e empurra o modelo a superotimizar as preferencias do dataset, arriscando degradar fluencia, cobertura e estabilidade; um `beta` muito baixo torna a mudanca quase imperceptivel. Por isso, `beta = 0.1` funciona como um "imposto" sobre desvios grandes da politica original: ele ainda recompensa respostas mais seguras, mas mantem o custo matematico de fugir demais do comportamento linguistico aprendido pelo modelo de referencia, preservando melhor a qualidade geral da linguagem.

## Hiperparametros e economia de memoria

Configuracoes principais do treino:

- `beta=0.1`
- `optim="paged_adamw_32bit"`
- `gradient_checkpointing=True`
- `warmup_steps=0`
- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.float16`

## Execucao

### Requisitos

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Recriar o dataset de preferencias

```powershell
python generate_dpo_dataset.py
```

### Smoke test de treino

```powershell
python train_dpo.py --dataset-path dataset/hhh_preferences.jsonl --output-dir adapters/lab8-dpo-smoke --num-train-epochs 0.05 --save-steps 5 --logging-steps 1
```

### Execucao padrao

```powershell
python train_dpo.py --dataset-path dataset/hhh_preferences.jsonl --output-dir adapters/lab8-dpo
```

## Resultado validado

Execucao final validada em `19/04/2026`:

- `train_loss`: `0.6724`
- `eval_loss`: `0.6151`
- `Chosen avg logprob`: `-4.1752`
- `Rejected avg logprob`: `-4.2956`
- `Diferenca (chosen - rejected)`: `0.1204`
- `Preferencia segura reforcada`: `True`

Esses numeros mostram que, para o prompt de validacao, a resposta segura passou a receber maior probabilidade media que a resposta rejeitada, atendendo ao criterio central do laboratorio.

## Validacao

Ao final do treino, o script:

- escolhe um exemplo de validacao do dataset
- calcula a log-probabilidade media da resposta `chosen`
- calcula a log-probabilidade media da resposta `rejected`
- gera uma resposta livre do modelo alinhado para o mesmo prompt
- salva um resumo estruturado em `reports/lab8_validation_summary.json`

O comportamento esperado no console e observar que a resposta segura recebe score melhor que a resposta rejeitada, mostrando que a preferencia foi internalizada.

## Checklist para correcao

- dataset com colunas estritamente `prompt`, `chosen` e `rejected`
- uso de `DPOTrainer` com `beta=0.1`
- explicacao matematica do papel de `beta` presente neste `README.md`
- treinamento executado com economia de memoria
- validacao console-based demonstrando preferencia da resposta segura
- nota obrigatoria de uso de IA incluida ao final deste documento

## Observacoes

- O script exige CUDA disponivel para o treino quantizado com `bitsandbytes`.
- O modelo padrao continua sendo `HuggingFaceTB/SmolLM-135M-Instruct`, escolhido para manter o laboratorio viavel em hardware academico.
- O adaptador do Lab 07 pode ser reaproveitado, mas o pipeline tambem funciona sem ele.
- A geracao livre apos o treino pode permanecer simples ou repetitiva por causa do porte reduzido do modelo base. A validacao principal da entrega e a comparacao objetiva entre `chosen` e `rejected`.

## Uso de IA

Partes geradas/complementadas com IA, revisadas por Carlos Eugenio.

## Versionamento

Versao final de entrega: `v1.0`
