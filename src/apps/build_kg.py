import os

from kg_gen import KGGen
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
#os.environ['XINFERENCE_API_BASE'] = "https://u445311-8282-9a8f85fc.yza1.seetacloud.com:8443/v1"
#os.environ['XINFERENCE_API_KEY'] = "anything" #[optional] no api key required

print(os.getenv('OPENAI_BASE_URL'))

#llm_model = 'xinference/deepseek-r1-distill-qwen'
#llm_model = 'openai/deepseek-r1-distill-qwen'
llm_model = 'deepseek/deepseek-chat'
# Initialize KGGen with optional configuration
kg = KGGen(
  #model="openai/gpt-4o",  # Default model
  model=llm_model,
  temperature=0.0,        # Default temperature
  #api_key="anything", # Optional if set in environment or using a local model
)

# EXAMPLE 1: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1 = kg.generate(
  input_data=text_input,
  context="Family relationships"
)
# Output:
# entities={'Linda', 'Ben', 'Andrew', 'Josh'}
# edges={'is brother of', 'is father of', 'is mother of'}
# relations={('Ben', 'is brother of', 'Josh'),
#           ('Andrew', 'is father of', 'Josh'),
#           ('Linda', 'is mother of', 'Josh')}

print(graph_1)

text_input = "第十条  办理票据回购方式的再贴现业务时，票据的回购期限自中国人民银行批准给付票据回购资金之日起，至公司与中国人民银行约定的票据回购日（不得为法定节假日），最长不得超过汇票到期日前7天。公司贴现5天后方可向中国人民银行申请票据再贴现。"
graph_1 = kg.generate(
  input_data=text_input,
  context="Financial rules"
)
print(graph_1)

text_input = """
{
    "function_name": "aeExportNetlistCDLShow",
    "function_definition": "aeExportNetlistCDLShow()",
    "parameters": [],
    "returns": {
        "type": "outbool",
        "description": "Always return true."
    },
    "examples": "import pyAether as pyScript\npyScript.emyInitAether(\"-ADV\")\npyScript.aeExportNetlistCDLShow()",
    "function_description": "This command pops up the Export Netlist window and switches to the CDL option."
}"""
graph_1 = kg.generate(
  input_data=text_input,
  context="Python API definition"
)
print(graph_1)
