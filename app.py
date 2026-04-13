import chainlit as cl
import os
import shutil  
from pipeline import run_full_pipeline 
from agent import agent_executor   

@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! Please upload a **PDF** or **Image** to begin.").send()

@cl.on_message
async def main(message: cl.Message):
    # 1. Handle File Uploads
    if message.elements:
        for element in message.elements:
            if element.type == "file" or element.type == "image":
                if not os.path.exists("uploads"):
                    os.makedirs("uploads")
                
                file_path = f"uploads/{element.name}"
                
                
                # If the file is large, Chainlit saves it to element.path. 
                # Otherwise, it's in element.content.
                if hasattr(element, "path") and element.path:
                    shutil.copy(element.path, file_path)
                else:
                    with open(file_path, "wb") as f:
                        f.write(element.content)
                

                msg = cl.Message(content=f"Processing `{element.name}`... this might take a while for large PDFs!")
                await msg.send()

                try:
                    run_full_pipeline(file_path)
                    await cl.Message(content=f"✅ `{element.name}` ingested! Ask me anything.").send()
                except Exception as e:
                    await cl.Message(content=f"❌ Error: {e}").send()
        return

    # 2. Handle Text Questions
    ui_message = cl.Message(content="")
    await ui_message.send()

    async with cl.Step(name="Agentic Router") as step:
        step.input = message.content
        for event in agent_executor.stream({"messages": [("user", message.content)]}):
            for node_name, node_state in event.items():
                if node_name == "tools":
                    step.output = "Consulting databases..."
                elif node_name == "agent":
                    ui_message.content = node_state['messages'][0].content
    
    await ui_message.update()