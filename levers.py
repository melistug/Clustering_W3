import pandas as pd

# model_doc.csv yolunu ver
MODEL_DOC_PATH = r"C:\Users\ASUS\Desktop\world3_stage1_full\model_doc.csv"

df = pd.read_csv(MODEL_DOC_PATH)

# Kolon isimlerini normalize edelim
df.columns = [c.strip() for c in df.columns]


def classify(row):
    t = str(row["Type"]).strip()
    st = str(row["Subtype"]).strip()

    if t == "Constant" and st == "Normal":
        return "SAFE (oynanabilir)"

    if t == "Auxiliary" and st == "Normal":
        return "RISKLI (önerilmez)"

    return "DO NOT TOUCH (modeli bozar)"


df["CALIBRATION_STATUS"] = df.apply(classify, axis=1)

# Özet
print("\n📊 ÖZET:")
print(df["CALIBRATION_STATUS"].value_counts())

# Ayrı ayrı listeler
safe = df[df["CALIBRATION_STATUS"].str.contains("SAFE")]
risky = df[df["CALIBRATION_STATUS"].str.contains("RISKLI")]
danger = df[df["CALIBRATION_STATUS"].str.contains("DO NOT TOUCH")]

# Kaydet
safe.to_csv("playable_SAFE.csv", index=False)
risky.to_csv("playable_RISKY.csv", index=False)
danger.to_csv("playable_DO_NOT_TOUCH.csv", index=False)

print("\n✅ Dosyalar yazıldı:")
print(" - playable_SAFE.csv")
print(" - playable_RISKY.csv")
print(" - playable_DO_NOT_TOUCH.csv")
