import pandas as pd
print("\n🔍 Training Data Info:")
MCP_Train_DF = pd.read_csv(r"Raw_Data\train.csv")
MCP_Test_DF = pd.read_csv(r"Raw_Data\test.csv")

print(f"\n📊 Training data shape: {MCP_Train_DF.shape}")
print(f"📈 Test data shape: {MCP_Test_DF.shape}")

# Display basic information
print("\n🔍 Training Data Info:")
print(MCP_Train_DF.info())

print("\n📋 First few rows:")
print(MCP_Train_DF.head())