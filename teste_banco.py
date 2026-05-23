import os
from dotenv import load_dotenv
from rag import get_rag

# Carrega o ambiente de forma idêntica ao app
load_dotenv()

print("--- Iniciando teste de conexão e gravação no Neo4j ---")
print(f"URI: {os.getenv('NEO4J_URI')}")
print(f"DB configurado: {os.getenv('NEO4J_DATABASE')}")

try:
    rag = get_rag()
    print("\n[1/3] Conexão e Schema inicializados com sucesso.")

    # Tentativa de gravação direta
    print("[2/3] Tentando indexar um texto de teste...")
    retonro_chunks = rag.index_text(
        text="O Banco possui soluções de crédito habitação com taxas mistas.",
        source="https://www.banco.pt/teste-conexao",
        kind="web"
    )
    print(f"-> Sucesso! Chunks gerados e enviados: {retonro_chunks}")

    # Consulta de validação imediata
    print("[3/3] Consultando contagem total de Chunks no banco...")
    total = rag.count()
    print(f"-> Total de chunks encontrados no Neo4j: {total}")

    if total == 0:
        print("\n[ALERTA] O driver não acusou erro, mas o Neo4j retornou 0 nós. Verifique se o Neo4j está rodando no modo 'ReadOnly' ou se o volume do Docker perdeu permissão de escrita.")

except Exception as e:
    print(f"\n[ERRO FATAL] A operação falhou: {e}")
    import traceback
    traceback.print_exc()