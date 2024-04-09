import gradio as gr
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from huggingface_hub import CommitScheduler

# Global variable to store the DataFrame
df = pd.read_csv('./feedback_results.csv')
df = df[['feedback','feedback_local','feedback_web']].rename(columns={'feedback':'none','feedback_local':'local','feedback_web':'web'})
col_names = list(df.columns)



# Setup file handling
JSON_DATASET_DIR = Path('json_dataset')
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)

JSON_DATASET_PATH = JSON_DATASET_DIR / f"{uuid4()}.json"

scheduler = CommitScheduler(
    repo_id='lancewilhelm/cs6804_final',
    repo_type='dataset',
    folder_path=JSON_DATASET_DIR,
    path_in_repo='data',
    every=1
)

def vote(vote, sample):
    """Cast a vote for the sample.

    Args:
        vote (str): the vote cast by the user
        sample (dict): the sample to vote on

    Returns:
        dict: the sample that was voted on
    """
    battle_outcome = {
        'sample_index': str(sample['index']),
        'sample_a_method': sample['a_method'],
        'sample_b_method': sample['b_method']
    }
    if vote == 'A':
        battle_outcome['winner'] = sample['a_method']
    elif vote == 'B':
        battle_outcome['winner'] = sample['b_method']
    elif vote == 'Tie':
        battle_outcome['winner'] = 'tie'
    # print(f"Voted: {sample['index']}")
    save_json(battle_outcome)
    # print(battle_outcome)
    return sample

def save_json(data: dict) -> None:
    """Save the data to a JSON file.

    Args:
        data (dict): the data to save
    """
    with scheduler.lock:
        with JSON_DATASET_PATH.open("a") as f:
            json.dump(data, f)
            f.write("\n")

# Main App
with gr.Blocks(theme=gr.themes.Soft()) as app:
    # Generate intial random feedback to show
    s = gr.State({})

    def get_random_feedback(sample):
        """Get a random sample of feedback.

        Args:
            sample (dict): the sample to populate

        Returns:
            dict: the sample of feedback
        """
        x = df.sample(1, replace=True, axis=0)
        sample['index'] = x.index[0]
        x = x.sample(2, replace=False, axis=1)
        sample['a_method'] = x.columns[0]
        sample['b_method'] = x.columns[1]
        sample['a_feedback'] = x.values[0][0]
        sample['b_feedback'] = x.values[0][1]
        # print(f"Sampled: {sample['index']}")
        return [sample, sample['a_feedback'], sample['b_feedback']]

    # Title and description
    gr.Markdown("## Feedback Battle")
    gr.Markdown("Below you will be provided with two pieces of feedback related to a hypothetical leadership communication utterance. Your task is to choose which of the two you feel is more **trustworthy**.")

    # First Row - More titles
    with gr.Row():
        gr.Markdown("### Feedback A")
        gr.Markdown("### Feedback B")

    # Second Row - Feedback
    with gr.Row():
        feedback_a = gr.Markdown(label='A')
        feedback_b = gr.Markdown(label='B')

    # Third Row - Vote Buttons
    with gr.Row():
        gr.Markdown('### Vote Which is More Trustworthy')
        btn_a = gr.Button("A")
        btn_b = gr.Button("B")
        btn_tie = gr.Button("Tie")

    # Functions of buttons
    btn_a.click(vote, inputs=[gr.State('A'), s], outputs=[s]).success(get_random_feedback, inputs=[s], outputs=[s, feedback_a, feedback_b])
    btn_b.click(vote, inputs=[gr.State('B'), s], outputs=[s]).success(get_random_feedback, inputs=[s], outputs=[s, feedback_a, feedback_b])
    btn_tie.click(vote, inputs=[gr.State('Tie'), s], outputs=[s]).success(get_random_feedback, inputs=[s], outputs=[s, feedback_a, feedback_b])

    # Initial loading function
    app.load(get_random_feedback, inputs=[s], outputs=[s, feedback_a, feedback_b])

app.launch()