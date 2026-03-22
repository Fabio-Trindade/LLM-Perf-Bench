import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Uso: python plot_temp.py <arquivo.csv>")
        sys.exit(1)

    arquivo = sys.argv[1]

    # Lê o arquivo (data/hora,temperatura)
    try:
        df = pd.read_csv(
            arquivo,
            sep=",",
            header=None,
            names=["datetime", "temperature"],
            parse_dates=["datetime"],
            date_parser=lambda x: pd.to_datetime(x, errors="coerce")
        )
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        sys.exit(1)

    # Remove linhas inválidas
    df = df.dropna(subset=["datetime", "temperature"])

    # Converte temperatura para numérico (em caso de vírgula decimal)
    df["temperature"] = df["temperature"].astype(str).str.replace(",", ".")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df = df.dropna(subset=["temperature"])

    if df.empty:
        print("Nenhum dado válido encontrado.")
        sys.exit(1)

    # Gera o gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(df["datetime"], df["temperature"], marker="o", linestyle="-")
    plt.title("Temperatura ao longo do tempo")
    plt.xlabel("Data/Hora")
    plt.ylabel("Temperatura (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("img.png")

if __name__ == "__main__":
    main()
