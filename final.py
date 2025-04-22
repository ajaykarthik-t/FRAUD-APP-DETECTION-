import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import time
import matplotlib.pyplot as plt
import random
import hashlib
from datetime import datetime
import csv

# File paths for CSV storage
REPORTS_CSV = 'reports.csv'
ADMIN_USERS_CSV = 'admin_users.csv'

# Initialize the CSV files if they don't exist
def init_csv_files():
    # Create reports.csv if it doesn't exist
    if not os.path.exists(REPORTS_CSV):
        with open(REPORTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'ticket_id', 'report_type', 'severity', 'title', 
                           'description', 'contact_email', 'status', 'created_at'])
    
    # Create admin_users.csv if it doesn't exist
    if not os.path.exists(ADMIN_USERS_CSV):
        with open(ADMIN_USERS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'username', 'password_hash', 'last_login'])
            # Add default admin user
            default_password = "admin123"
            password_hash = hashlib.sha256(default_password.encode()).hexdigest()
            writer.writerow([1, 'admin', password_hash, ''])

# Get next available ID for a CSV file
def get_next_id(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return 1
        return df['id'].max() + 1
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 1

# Save report to CSV
def save_report(report_type, severity, title, description, contact_email):
    # Generate a unique ticket ID
    ticket_id = f"SR-{random.randint(10000, 99999)}"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get next ID
    next_id = get_next_id(REPORTS_CSV)
    
    # Append to CSV
    with open(REPORTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            next_id, ticket_id, report_type, severity, title, 
            description, contact_email, 'Pending', created_at
        ])
    
    return ticket_id

# Get a specific report by ticket ID
def get_report_by_ticket(ticket_id):
    try:
        df = pd.read_csv(REPORTS_CSV)
        report = df[df['ticket_id'] == ticket_id]
        if not report.empty:
            return report.iloc[0]
        return None
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

# Get all reports for admin panel
def get_all_reports():
    try:
        df = pd.read_csv(REPORTS_CSV)
        # Ensure the status column contains proper status values
        valid_status_values = ["Pending", "In Progress", "Resolved", "Closed"]
        
        # Check and fix status values if needed
        for i, status in enumerate(df['status']):
            if status not in valid_status_values:
                df.at[i, 'status'] = 'Pending'  # Default to 'Pending' if invalid
        
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'id', 'ticket_id', 'report_type', 'severity', 'title', 
            'description', 'contact_email', 'status', 'created_at'
        ])

# Update report status
def update_report_status(ticket_id, new_status):
    try:
        df = pd.read_csv(REPORTS_CSV)
        if ticket_id in df['ticket_id'].values:
            df.loc[df['ticket_id'] == ticket_id, 'status'] = new_status
            df.to_csv(REPORTS_CSV, index=False)
            return True
        return False
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False
    except Exception as e:
        st.error(f"Error updating status: {str(e)}")
        return False

# Verify admin login
def verify_admin(username, password):
    try:
        df = pd.read_csv(ADMIN_USERS_CSV)
        user_row = df[df['username'] == username]
        
        if not user_row.empty:
            stored_hash = user_row['password_hash'].values[0]
            # Hash the provided password and compare
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash == stored_hash:
                # Update last login time
                df.loc[df['username'] == username, 'last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.to_csv(ADMIN_USERS_CSV, index=False)
                return True
        return False
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False

# Function to safely format dates, handling NaT values
def safe_date_format(date_val, format_str="%Y-%m-%d %H:%M"):
    """Safely format a date value, returning a fallback string if the date is NaT or invalid"""
    try:
        if pd.isna(date_val):
            return "Date unknown"
        return pd.to_datetime(date_val).strftime(format_str)
    except:
        return "Date unknown"

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F0F2F6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .prediction-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .severity-high { background-color: #FFCCCB; border-left: 5px solid #FF0000; }
    .severity-medium { background-color: #FFE4B5; border-left: 5px solid #FFA500; }
    .severity-low { background-color: #90EE90; border-left: 5px solid #008000; }
    .severity-critical { background-color: #FF6B6B; border-left: 5px solid #8B0000; }
    
    .report-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    .admin-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E4053;
        margin: 0;
    }
    
    .submit-button {
        background-color: #9B59B6 !important;
    }
    
    .admin-button {
        background-color: #E74C3C !important;
    }
    
    /* Tables styling */
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 10px;
        overflow: hidden;
    }
    .styled-table thead tr {
        background-color: #3498DB;
        color: white;
        text-align: left;
    }
    .styled-table th, .styled-table td {
        padding: 12px 15px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #3498DB;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
        text-align: center;
        margin: 5px 0;
    }
    
    .status-badge-pending {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .status-badge-in-progress {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-badge-resolved {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-badge-closed {
        background-color: #e2e3e5;
        color: #383d41;
    }
    
    .ticket-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #6c757d;
        transition: all 0.3s ease;
    }
    
    .ticket-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .ticket-card.severity-low {
        border-left-color: #28a745;
    }
    
    .ticket-card.severity-medium {
        border-left-color: #ffc107;
    }
    
    .ticket-card.severity-high {
        border-left-color: #fd7e14;
    }
    
    .ticket-card.severity-critical {
        border-left-color: #dc3545;
    }
    
    .ticket-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .ticket-id {
        font-weight: bold;
        color: #495057;
        font-size: 1.1em;
    }
    
    .ticket-title {
        font-size: 1.2em;
        font-weight: bold;
        margin: 10px 0;
        color: #212529;
    }
    
    .ticket-meta {
        display: flex;
        gap: 15px;
        margin: 10px 0;
        color: #6c757d;
        font-size: 0.9em;
    }
    
    .ticket-meta-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .ticket-description {
        margin: 15px 0;
        color: #495057;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Load model and preprocessing components"""
    try:
        model = load_model('attack_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, tokenizer, encoders
    except:
        # Return dummy components for demo purposes if files aren't available
        return None, None, None

def create_description(row):
    """Recreate the description from CSV row"""
    return (f"Incident {row['incident_id']} of {row['severity']} severity involved "
            f"primary tactic {row['primary_tactic']} using technique {row['technique_id']}. "
            f"Secondary tactics observed: {row['secondary_tactics']} "
            f"with techniques: {row['secondary_techniques']}")

def preprocess_input(df, tokenizer, max_len=250):
    """Preprocess the input data"""
    descriptions = df.apply(create_description, axis=1).str.lower()
    sequences = tokenizer.texts_to_sequences(descriptions)
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

def get_status_badge(status):
    """Return the appropriate status badge HTML"""
    # Ensure we only deal with valid status values
    valid_statuses = ["Pending", "In Progress", "Resolved", "Closed"]
    if status not in valid_statuses:
        status = "Pending"  # Default to Pending for any unknown status
    
    status_class = status.lower().replace(" ", "-")
    return f'<span class="status-badge status-badge-{status_class}">{status}</span>'

def show_admin_panel():
    """Display admin panel for report management"""
    st.markdown("### 👨‍💼 Admin Panel")
    
    # Display only Reports tab - removed Scan History and Threats tabs
    st.subheader("📋 User Reports")
    
    try:
        reports_df = get_all_reports()
        if not reports_df.empty:
            # Format dates for better display - safely handle NaT values
            reports_df['created_at'] = pd.to_datetime(reports_df['created_at'], errors='coerce')
            reports_df['formatted_date'] = reports_df['created_at'].apply(safe_date_format)
            
            # Display reports in a table
            st.dataframe(
                reports_df[['ticket_id', 'report_type', 'severity', 'title', 'formatted_date', 'status']],
                use_container_width=True
            )
            
            # Report details
            st.subheader("Report Details")
            try:
                selected_ticket = st.selectbox("Select a ticket ID to view details:", 
                                            reports_df['ticket_id'].tolist())
                
                if selected_ticket:
                    report = reports_df[reports_df['ticket_id'] == selected_ticket].iloc[0]
                    
                    # Make sure the status is valid
                    valid_statuses = ["Pending", "In Progress", "Resolved", "Closed"]
                    current_status = report['status']
                    if current_status not in valid_statuses:
                        current_status = "Pending"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Ticket ID:** {report['ticket_id']}")
                        st.markdown(f"**Report Type:** {report['report_type']}")
                        st.markdown(f"**Severity:** {report['severity']}")
                    with col2:
                        st.markdown(f"**Status:** {current_status}")
                        st.markdown(f"**Submitted:** {report['formatted_date']}")
                        st.markdown(f"**Contact:** {report['contact_email']}")
                    
                    st.markdown(f"**Title:** {report['title']}")
                    st.markdown("**Description:**")
                    st.text_area("", value=report['description'], height=150, disabled=True)
                    
                    # Status update
                    status_options = ["Pending", "In Progress", "Resolved", "Closed"]
                    try:
                        current_index = status_options.index(current_status)
                    except ValueError:
                        current_index = 0  # Default to Pending if status is not valid
                        
                    new_status = st.selectbox("Update Status:", 
                                            status_options,
                                            index=current_index)
                    
                    if st.button("Update Status"):
                        try:
                            if update_report_status(selected_ticket, new_status):
                                st.success(f"Status updated to {new_status}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to update status")
                        except Exception as e:
                            st.error(f"Error updating status: {str(e)}")
            except Exception as e:
                st.error(f"Error loading report details: {str(e)}")
        else:
            st.info("No reports have been submitted yet.")
    except Exception as e:
        st.error(f"Error retrieving reports: {str(e)}")

def main():
    # Initialize the CSV files
    init_csv_files()
    
    # Session state initialization
    if 'admin_view' not in st.session_state:
        st.session_state.admin_view = False
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    if 'show_report_form' not in st.session_state:
        st.session_state.show_report_form = True
    if 'last_submitted_ticket' not in st.session_state:
        st.session_state.last_submitted_ticket = None
    
    # Main application header
    st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">🛡️ Cyber Security Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Admin login in sidebar
    with st.sidebar:
        st.title("🔐 Control Panel")
        
        # Admin login section
        if not st.session_state.admin_authenticated:
            st.subheader("Admin Login")
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if verify_admin(username, password):
                    st.session_state.admin_authenticated = True
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid credentials!")
        else:
            st.success("Logged in as Admin")
            if st.button("View Admin Panel"):
                st.session_state.admin_view = True
                st.rerun()
            if st.button("Back to User Dashboard"):
                st.session_state.admin_view = False
                st.rerun()
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.session_state.admin_view = False
                st.rerun()
    
    # Admin view
    if st.session_state.admin_view and st.session_state.admin_authenticated:
        show_admin_panel()
    else:
        # Regular user interface with tabs
        tabs = st.tabs(["🔍 Attack Predictor", "📝 Issues & Support"])
        
        # 1. Attack Predictor Tab
        with tabs[0]:
            st.markdown("### Upload incident data CSV to predict attack techniques and tactics")
            
            # Sidebar for file upload and info
            with st.sidebar:
                st.header("⚙️ Configuration")
                uploaded_file = st.file_uploader("Upload CSV", type="csv")
                model, tokenizer, encoders = load_components()
                
                st.markdown("---")
                st.markdown("**Expected CSV Columns:**")
                st.write("- incident_id, severity, primary_tactic")
                st.write("- technique_id, secondary_tactics")
                st.write("- secondary_techniques")

            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    required_columns = ['incident_id', 'severity', 'primary_tactic',
                                    'technique_id', 'secondary_tactics', 'secondary_techniques']
                    
                    if not all(col in df.columns for col in required_columns):
                        st.error("❌ Invalid CSV format. Missing required columns.")
                        return
                    
                    # Create dummy predictions if model is not available
                    if model is None:
                        st.info("📝 Demo mode: Using simulated predictions")
                        predictions = np.random.rand(len(df), 10)
                        predicted_indices = np.argmax(predictions, axis=1)
                        confidence = np.max(predictions, axis=1)
                        technique_options = ["T1001", "T1002", "T1003", "T1004", "T1005", "T1006", "T1007", "T1008", "T1009", "T1010"]
                        predicted_techniques = [technique_options[i] for i in predicted_indices]
                    else:
                        # Preprocess data
                        X = preprocess_input(df, tokenizer)
                        
                        # Make predictions
                        predictions = model.predict(X)
                        predicted_indices = np.argmax(predictions, axis=1)
                        confidence = np.max(predictions, axis=1)
                        predicted_techniques = encoders['technique_encoder'].inverse_transform(predicted_indices)
                    
                    # Display results
                    st.subheader("🔍 Prediction Results")
                    for idx, row in df.iterrows():
                        with st.container():
                            # Severity color coding
                            severity_class = f"severity-{row['severity'].lower()}"
                            st.markdown(f"""
                                <div class="prediction-box {severity_class}">
                                    <h4>Incident ID: {row['incident_id']}</h4>
                                    <div style="display: flex; gap: 20px;">
                                        <div style="flex: 1;">
                                            <p>📊 Severity: <strong>{row['severity']}</strong></p>
                                            <p>🔑 Primary Tactic: {row['primary_tactic']}</p>
                                        </div>
                                        <div style="flex: 1;">
                                            <p>🎯 Predicted Technique: <strong>{predicted_techniques[idx]}</strong></p>
                                            <p>✅ Confidence: {confidence[idx]:.1%}</p>
                                        </div>
                                    </div>
                                    <progress value="{confidence[idx]}" max="1" style="width: 100%;"></progress>
                                </div>
                            """, unsafe_allow_html=True)

                    # Show summary statistics
                    st.markdown("---")
                    st.subheader("📈 Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Incidents", len(df))
                    with col2:
                        st.metric("Most Common Technique", pd.Series(predicted_techniques).mode()[0])
                    with col3:
                        avg_confidence = np.mean(confidence)
                        st.metric("Average Confidence", f"{avg_confidence:.1%}")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        # 2. Issues & Support Tab
        with tabs[1]:
            report_container = st.container()
            
            # Create tabs for "Submit Issue" and "Check Status"
            issue_tabs = st.tabs(["📮 Submit Issue", "🔍 Check Status"])
            
            # Submit Issue Tab
            with issue_tabs[0]:
                st.markdown("""
                    <div class="report-container">
                        <h3>Submit an issue report to our security team</h3>
                        <p>If you're experiencing security problems or have concerns about potential threats, please fill out the form below to notify our team.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create a form for issue reporting
                with st.form("issue_report_form"):
                    # Form fields
                    report_type = st.selectbox(
                        "Issue Type *",
                        ["Select an issue type", "Suspected Malware", "System Performance", "False Positive", "Software Bug", "Other"]
                    )
                    
                    severity_level = st.select_slider(
                        "Severity Level *",
                        options=["Low", "Medium", "High", "Critical"],
                        value="Medium"
                    )
                    
                    issue_title = st.text_input("Issue Title *", placeholder="Brief description of the issue")
                    
                    issue_description = st.text_area(
                        "Detailed Description *",
                        placeholder="Please provide as much detail as possible about the issue you're experiencing...",
                        height=150
                    )
                    
                    contact_info = st.text_input("Contact Email (for updates) *", placeholder="your.email@example.com")
                    
                    # Information about follow-up
                    st.info("""
                    Your report will be reviewed by our security team. Depending on the severity and complexity of the issue, 
                    you can expect a response within 24-48 hours via the contact email you provide.
                    """)
                    
                    # Submit button
                    submit_button = st.form_submit_button("📤 Submit Report", use_container_width=True)
                    
                    if submit_button:
                        if report_type == "Select an issue type" or not issue_title or not issue_description or not contact_info:
                            st.error("Please fill out all required fields marked with *")
                        else:
                            # Save report to CSV
                            ticket_id = save_report(
                                report_type, 
                                severity_level, 
                                issue_title, 
                                issue_description, 
                                contact_info
                            )
                            
                            # Store the ticket ID for use in the Check Status tab
                            st.session_state.last_submitted_ticket = ticket_id
                            
                            # Success message with ticket ID
                            st.success(f"Thank you! Your report has been submitted successfully. Your ticket ID is {ticket_id}.")
                            
                            # Display confirmation details
                            st.info(f"""
                            We have received your report regarding "{issue_title}".
                            Our security team will review your report and respond within 24-48 hours
                            to the email address you provided ({contact_info}).
                            
                            Please note your ticket ID ({ticket_id}) for checking the status of your report.
                            """)
            
            # Check Status Tab
            with issue_tabs[1]:
                st.markdown("""
                    <div class="report-container">
                        <h3>Check the status of your submitted issues</h3>
                        <p>Enter your ticket ID to check the current status or view recent submissions.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Two ways to check: direct ticket lookup or show recent issues
                check_cols = st.columns([2, 1])
                
                with check_cols[0]:
                    ticket_id_input = st.text_input("Enter Ticket ID", 
                                                 value=st.session_state.last_submitted_ticket if st.session_state.last_submitted_ticket else "",
                                                 placeholder="e.g., SR-12345")
                
                with check_cols[1]:
                    check_button = st.button("Check Status", use_container_width=True)
                
                # Show specific ticket details if requested
                if check_button and ticket_id_input:
                    report = get_report_by_ticket(ticket_id_input)
                    
                    if report is not None:
                        # Format date safely
                        created_at = pd.to_datetime(report['created_at'], errors='coerce')
                        formatted_date = safe_date_format(created_at)
                        
                        # Ensure status is valid
                        status = report['status']
                        valid_statuses = ["Pending", "In Progress", "Resolved", "Closed"]
                        if status not in valid_statuses:
                            status = "Pending"
                        
                        # Status badge
                        status_html = get_status_badge(status)
                        
                        # Display ticket information in a styled card
                        st.markdown(f"""
                            <div class="ticket-card severity-{report['severity'].lower()}">
                                <div class="ticket-header">
                                    <span class="ticket-id">{report['ticket_id']}</span>
                                    {status_html}
                                </div>
                                <div class="ticket-title">{report['title']}</div>
                                <div class="ticket-meta">
                                    <div class="ticket-meta-item">
                                        <span>📅 Submitted:</span> {formatted_date}
                                    </div>
                                    <div class="ticket-meta-item">
                                        <span>🔥 Severity:</span> {report['severity']}
                                    </div>
                                    <div class="ticket-meta-item">
                                        <span>📋 Type:</span> {report['report_type']}
                                    </div>
                                </div>
                                <div class="ticket-description">{report['description']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Provide guidance based on status
                        if status == 'Pending':
                            st.info("Your report is pending review by our security team. Please allow up to 24-48 hours for an initial response.")
                        elif status == 'In Progress':
                            st.info("Our security team is actively working on your issue. You should receive updates via email.")
                        elif status == 'Resolved':
                            st.success("This issue has been resolved. If you're still experiencing problems, please submit a new report or contact support.")
                        elif status == 'Closed':
                            st.warning("This ticket has been closed. If you need further assistance, please submit a new report.")
                    else:
                        st.error(f"No report found with ticket ID: {ticket_id_input}")
                
                # Show recent submissions if available
                st.markdown("### Recent Submissions")
                
                try:
                    reports_df = get_all_reports()
                    if not reports_df.empty:
                        # Sort by creation date (newest first) and take the most recent 5
                        reports_df['created_at'] = pd.to_datetime(reports_df['created_at'], errors='coerce')
                        recent_reports = reports_df.sort_values('created_at', ascending=False).head(5)
                        
                        for _, report in recent_reports.iterrows():
                            # Format date safely
                            formatted_date = safe_date_format(report['created_at'])
                            
                            # Ensure status is valid
                            status = report['status']
                            valid_statuses = ["Pending", "In Progress", "Resolved", "Closed"]
                            if status not in valid_statuses:
                                status = "Pending"
                            
                            # Status badge
                            status_html = get_status_badge(status)
                            
                            # Display ticket card
                            st.markdown(f"""
                                <div class="ticket-card severity-{report['severity'].lower()}">
                                    <div class="ticket-header">
                                        <span class="ticket-id">{report['ticket_id']}</span>
                                        {status_html}
                                    </div>
                                    <div class="ticket-title">{report['title']}</div>
                                    <div class="ticket-meta">
                                        <div class="ticket-meta-item">
                                            <span>📅 Submitted:</span> {formatted_date}
                                        </div>
                                        <div class="ticket-meta-item">
                                            <span>🔥 Severity:</span> {report['severity']}
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No reports have been submitted yet.")
                except Exception as e:
                    st.error(f"Error retrieving reports: {str(e)}")

if __name__ == "__main__":
    main()