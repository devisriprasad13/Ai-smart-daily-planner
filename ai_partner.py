# ai_daily_planner_enhanced.py
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import re
import dateparser

# ---------- Database ----------
conn = sqlite3.connect("tasks_enhanced.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    category TEXT,
    priority INTEGER,
    deadline TEXT,
    estimated_minutes INTEGER,
    created_at TEXT,
    completed_at TEXT,
    status TEXT DEFAULT 'Pending'
)
''')
conn.commit()

# ---------- Utilities ----------
def parse_natural_input(nl_text):
    """
    Very simple natural language parser:
      - finds a date/time using dateparser.search.search_dates
      - if found, removes the date phrase from title and returns parsed date
      - else returns None for deadline
    """
    if not nl_text or nl_text.strip() == "":
        return None, None

    # try to find dates using dateparser
    # dateparser can parse phrases like 'tomorrow at 3pm', 'next monday', '2025-10-21 14:00' etc.
    try:
        parsed = dateparser.parse(nl_text, settings={'PREFER_DATES_FROM': 'future'})
    except Exception:
        parsed = None

    if parsed:
        # remove common date/time words to create a cleaned title
        # a heuristic: remove anything that looks like a time or date (digits, am/pm, tomorrow, next, monday, etc.)
        cleaned = re.sub(r'\b(today|tomorrow|next|at|on|in|\bmon(day)?\b|\btue(sday)?\b|\bwed(nesday)?\b|\bthu(rsday)?\b|\bfri(day)?\b|\bsat(urday)?\b|\bsun(day)?\b)\b', '', nl_text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b\d{1,4}[:/.-]\d{1,4}\b', '', cleaned)  # remove simple date/time patterns
        cleaned = re.sub(r'\b\d{1,2}(:\d{2})?\s*(am|pm)?\b', '', cleaned, flags=re.IGNORECASE)  # remove times
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip(" -,:")
        title = cleaned if cleaned else nl_text
        return title.strip(), parsed
    else:
        return nl_text.strip(), None

def add_task_to_db(title, category, priority, deadline, estimated_minutes):
    created = datetime.now().isoformat()
    deadline_str = deadline.isoformat() if isinstance(deadline, (datetime, date)) else (str(deadline) if deadline else None)
    c.execute('''
        INSERT INTO tasks (title, category, priority, deadline, estimated_minutes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (title, category, priority, deadline_str, estimated_minutes, created))
    conn.commit()

def get_tasks_df():
    df = pd.read_sql_query("SELECT * FROM tasks", conn, parse_dates=['created_at', 'completed_at', 'deadline'])
    # ensure deadline parsed where possible
    try:
        df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
    except Exception:
        pass
    return df

def mark_complete(task_id):
    now = datetime.now().isoformat()
    c.execute("UPDATE tasks SET status='Completed', completed_at=?, WHERECLAUSE = ? WHERE id = ?")
    # Some SQLite versions may be strict about column names; safer to use separate update
    # We'll do two-step update to avoid syntax issues:
    c.execute("UPDATE tasks SET status = 'Completed' WHERE id = ?", (task_id,))
    c.execute("UPDATE tasks SET completed_at = ? WHERE id = ?", (now, task_id))
    conn.commit()

def delete_task(task_id):
    c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()

def update_task_status(task_id, new_status):
    if new_status == 'Completed':
        now = datetime.now().isoformat()
        c.execute("UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?", (new_status, now, task_id))
    else:
        c.execute("UPDATE tasks SET status = ? WHERE id = ?", (new_status, task_id))
    conn.commit()

# ---------- AI scheduler / duration prediction ----------
def ai_schedule(df):
    """
    Simple "AI" schedule:
      - sort by status (pending first), then priority desc, then nearest deadline
      - include predicted duration: if estimated_minutes is null, use average of same category, else global average
    """
    if df.empty:
        return df
    df = df.copy()
    df['deadline_parsed'] = pd.to_datetime(df['deadline'], errors='coerce')
    # Predicted duration
    df['pred_minutes'] = df['estimated_minutes']
    # compute category averages
    cat_avgs = df[df['estimated_minutes'].notnull()].groupby('category')['estimated_minutes'].mean().to_dict()
    global_avg = df['estimated_minutes'].mean()
    for idx, row in df.iterrows():
        if pd.isna(row['pred_minutes']):
            cat = row['category']
            if cat in cat_avgs and not np.isnan(cat_avgs[cat]):
                df.at[idx, 'pred_minutes'] = int(round(cat_avgs[cat]))
            elif not np.isnan(global_avg):
                df.at[idx, 'pred_minutes'] = int(round(global_avg))
            else:
                df.at[idx, 'pred_minutes'] = 30  # fallback 30 minutes
    # sort
    df['status_rank'] = df['status'].apply(lambda s: 0 if s == 'Pending' else 1)
    df_sorted = df.sort_values(by=['status_rank', 'priority', 'deadline_parsed'], ascending=[True, False, True])
    return df_sorted

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Smart Daily Planner (Enhanced)", layout="wide")
st.title("🧭 AI Smart Daily Planner — Enhanced")

sidebar_choice = st.sidebar.radio("Go to", ["Add Task (NLP)", "All Tasks", "AI Suggested Schedule", "Analytics", "Settings"])

# ---------- Add Task (NLP) ----------
if sidebar_choice == "Add Task (NLP)":
    st.header("Add Task — Natural Language Input")
    st.write("Type naturally, e.g. 'Meeting with Ravi tomorrow at 3pm for 45 minutes' or 'Submit report next monday'")

    nl_input = st.text_area("Describe the task (natural language)", placeholder="e.g. Call mom tomorrow 7pm, Weekly sync next Monday 9am")
    st.write("Or fill fields below if you prefer explicit entry:")

    # explicit fields
    col1, col2, col3 = st.columns(3)
    with col1:
        title_input = st.text_input("Title (explicit)")
        category = st.selectbox("Category", ["Work", "Personal", "Study", "Fitness", "Others"])
    with col2:
        priority = st.slider("Priority (1 = low, 5 = high)", 1, 5, 3)
        deadline_input = st.date_input("Deadline (explicit)", value=None)
    with col3:
        est_minutes = st.number_input("Estimated minutes (optional)", min_value=0, step=5, value=0)
        est_minutes = int(est_minutes) if est_minutes > 0 else None

    if st.button("Add Task"):
        # try NLP parse first if provided
        if nl_input and nl_input.strip():
            parsed_title, parsed_dt = parse_natural_input(nl_input)
            final_title = parsed_title if parsed_title else (title_input if title_input else "Untitled Task")
            final_deadline = parsed_dt if parsed_dt else (deadline_input if deadline_input else None)
            # try to extract duration from text using regex like 'for 45 minutes', '45 mins', '1.5 hr'
            est_from_text = None
            m = re.search(r'(\d+)\s*(minutes|minute|mins|min)\b', nl_input, flags=re.IGNORECASE)
            if not m:
                m = re.search(r'(\d+(\.\d+)?)\s*(hours|hour|hrs|hr)\b', nl_input, flags=re.IGNORECASE)
                if m:
                    hours = float(m.group(1))
                    est_from_text = int(round(hours*60))
            else:
                est_from_text = int(m.group(1))
            final_est = est_minutes or est_from_text
            add_task_to_db(final_title, category, priority, final_deadline, final_est)
            st.success(f"Task added: {final_title} — deadline: {final_deadline} — est: {final_est} min")
        else:
            final_title = title_input if title_input else "Untitled Task"
            final_deadline = deadline_input if deadline_input else None
            add_task_to_db(final_title, category, priority, final_deadline, est_minutes)
            st.success(f"Task added: {final_title}")

# ---------- View & Manage Tasks ----------
elif sidebar_choice == "All Tasks":
    st.header("All Tasks — Manage")
    df = get_tasks_df()
    if df.empty:
        st.info("No tasks yet. Add a new task from the left panel.")
    else:
        # show table
        st.dataframe(df.sort_values(by=['status', 'priority'], ascending=[True, False]).reset_index(drop=True))

        st.markdown("---")
        st.subheader("Update / Delete Task")
        task_ids = df['id'].tolist()
        selected = st.selectbox("Select task id", options=task_ids)
        trow = df[df['id'] == selected].iloc[0]
        st.write("**Title:**", trow['title'])
        st.write("**Category:**", trow['category'], " — **Priority:**", int(trow['priority']))
        st.write("**Status:**", trow['status'])
        cols = st.columns(3)
        with cols[0]:
            if st.button("Toggle Complete"):
                new_status = 'Completed' if trow['status'] != 'Completed' else 'Pending'
                update_task_status(selected, new_status)
                st.rerun()
        with cols[1]:
            if st.button("Delete Task"):
                delete_task(selected)
                st.success("Deleted.")
                st.rerun()
        with cols[2]:
            if st.button("Set Priority 5"):
                c.execute("UPDATE tasks SET priority = ? WHERE id = ?", (5, selected)); conn.commit()
                st.rerun()

# ---------- AI Suggested Schedule ----------
elif sidebar_choice == "AI Suggested Schedule":
    st.header("AI Suggested Schedule")
    df = get_tasks_df()
    if df.empty:
        st.info("No tasks yet.")
    else:
        scheduled = ai_schedule(df)
        st.write("Suggested order (Pending tasks first, then by priority desc, then nearest deadline):")
        display_cols = ['id', 'title', 'category', 'priority', 'deadline', 'pred_minutes', 'status']
        st.dataframe(scheduled[display_cols].reset_index(drop=True))

        # Offer to export the schedule as CSV
        csv = scheduled[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button("Download schedule CSV", csv, "ai_schedule.csv", "text/csv")

# ---------- Analytics ----------
elif sidebar_choice == "Analytics":
    st.header("Analytics — Productivity Insights")
    df = get_tasks_df()
    if df.empty:
        st.info("No data to show yet.")
    else:
        # 1) Completion rate
        total = len(df)
        completed = len(df[df['status'] == 'Completed'])
        pending = total - completed
        col1, col2 = st.columns([1,2])
        with col1:
            st.subheader("Completion")
            st.metric("Completed / Total", f"{completed} / {total}", delta=None)

        # Pie chart: status distribution
        fig1, ax1 = plt.subplots()
        status_counts = df['status'].value_counts()
        ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.0f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        st.markdown("---")

        # 2) Tasks per category (bar)
        st.subheader("Tasks per Category")
        cat_counts = df['category'].fillna('Others').value_counts()
        fig2, ax2 = plt.subplots()
        ax2.bar(cat_counts.index.astype(str), cat_counts.values)
        ax2.set_ylabel("Number of tasks")
        ax2.set_xticklabels(cat_counts.index.astype(str), rotation=20, ha='right')
        st.pyplot(fig2)

        st.markdown("---")

        # 3) Completion timeline (tasks completed per day)
        st.subheader("Completions Over Time")
        df_completed = df[df['status'] == 'Completed'].copy()
        if not df_completed.empty:
            df_completed['completed_date'] = pd.to_datetime(df_completed['completed_at']).dt.date
            series = df_completed.groupby('completed_date').size()
            fig3, ax3 = plt.subplots()
            ax3.plot(series.index.astype(str), series.values, marker='o')
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Completions")
            ax3.set_xticklabels(series.index.astype(str), rotation=30, ha='right')
            st.pyplot(fig3)
        else:
            st.info("No completed tasks yet to show timeline.")

        st.markdown("---")

        # 4) Average estimated time vs actual (we only have estimated in this version)
        st.subheader("Estimated Task Duration Distribution")
        if df['estimated_minutes'].notnull().any():
            est = df['estimated_minutes'].dropna()
            fig4, ax4 = plt.subplots()
            ax4.hist(est, bins=8)
            ax4.set_xlabel("Estimated minutes")
            ax4.set_ylabel("Count")
            st.pyplot(fig4)
        else:
            st.info("No estimated durations provided yet.")

# ---------- Settings ----------
elif sidebar_choice == "Settings":
    st.header("Settings & Helper")
    st.write("This simple app stores data in `tasks_enhanced.db` (SQLite) in the same folder.")
    st.write("To reset (delete all tasks), click below:")
    if st.button("Reset / Delete All Tasks (DANGER)"):
        c.execute("DELETE FROM tasks")
        conn.commit()
        st.warning("All tasks deleted. Refresh to see changes.")
