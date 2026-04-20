# Evidencia de Execucao Final - Laboratorio 08

Data da execucao validada: 19/04/2026

Comando executado:

```powershell
.\venv\Scripts\python.exe train_dpo.py --dataset-path dataset/hhh_preferences.jsonl --output-dir adapters/lab8-dpo
```

Resultados observados:

- `train_loss`: `0.6724`
- `eval_loss`: `0.6151`
- `Chosen avg logprob`: `-4.1752`
- `Rejected avg logprob`: `-4.2956`
- `Diferenca (chosen - rejected)`: `0.1204`
- `Preferencia segura reforcada`: `True`

Interpretacao:

O treino DPO aumentou a preferencia relativa pela resposta segura (`chosen`) em comparacao com a resposta inadequada (`rejected`). Isso demonstra que o pipeline de alinhamento HHH foi executado com sucesso e atendeu ao criterio de validacao do laboratorio.
