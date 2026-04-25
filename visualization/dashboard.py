from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from training.evaluate import compare_agents


st.set_page_config(page_title="Email Triage RL Dashboard", layout="wide")


def flatten_agent_metrics(agent_name, result):
    agg = result.get("aggregated", {})
    row = {"agent": agent_name}

    for key, stats in agg.items():
        if isinstance(stats, dict) and "mean" in stats:
            row[key] = stats["mean"]

    return row


def main():
    st.title("Email Triage RL Benchmark Dashboard")
    st.markdown(
        """
        Multi-agent reinforcement learning environment for enterprise email triage with:
        - adversarial spam and phishing
        - SLA pressure
        - workload spikes
        - queue-based operational decisions
        """
    )

    st.info(
    "Self-improvement is tracked with a curriculum level derived from recent safety performance. "
    "The tracker recommends harder scenarios only after the agent is stable, avoiding PPO collapse."
    )

    st.sidebar.header("Evaluation Controls")
    num_episodes = st.sidebar.slider("Episodes per agent", 5, 50, 20, 5)
    checkpoint_path = st.sidebar.text_input(
        "PPO checkpoint path",
        value="training/checkpoints/ppo_email_triage.pt",
    )

    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation..."):
            comparison = compare_agents(
                num_episodes=num_episodes,
                checkpoint_path=checkpoint_path,
            )

        rows = [flatten_agent_metrics(agent, result) for agent, result in comparison.items()]
        df = pd.DataFrame(rows)

        st.subheader("Agent Comparison Table")
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Average Reward")
            st.bar_chart(df.set_index("agent")[["total_reward"]])

            st.subheader("Urgent Handled")
            st.bar_chart(df.set_index("agent")[["urgent_handled"]])

            st.subheader("Safe Policy Score")
            if "safe_policy_score" in df.columns:
                st.bar_chart(df.set_index("agent")[["safe_policy_score"]])

        with col2:
            st.subheader("Urgent Missed")
            st.bar_chart(df.set_index("agent")[["urgent_missed"]])

            st.subheader("SLA Breaches")
            st.bar_chart(df.set_index("agent")[["sla_breaches"]])

            if "cumulative_reward" in df.columns:
                st.subheader("Cumulative Reward Metric")
                st.bar_chart(df.set_index("agent")[["cumulative_reward"]])

        st.subheader("Interpretation")
        st.markdown(
            """
            - **Random** performs poorly across reward and safety.
            - **Rule-based** is efficient and strong on handcrafted heuristics.
            - **PPO** learns a safer operational policy, often reducing urgent misses and SLA breaches sharply.
            """
        )

        with st.expander("Raw comparison JSON"):
            st.json(comparison)
        


if __name__ == "__main__":
    main()