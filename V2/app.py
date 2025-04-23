import streamlit as st
from file_checker import checkFile

def main():
    st.title("Malware File Checker")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload a file to check for probable malware", type=["exe", "dll", "bin", "py", "txt", "pdf", "docx"])

    if uploaded_file is not None:
        # Get the file name
        filename = uploaded_file.name
        
        # Save the uploaded file temporarily
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Check if the file is legitimate
        legitimate = checkFile(filename)

        # Display the result
        if legitimate:
            st.success(f"The file '{filename}' seems to be legitimate.")
        else:
            st.error(f"The file '{filename}' is probably a MALWARE!!!")

if __name__ == "__main__":
    main()
