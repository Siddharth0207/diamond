try:
response = requests.post(f"http://localhost:8000/start_recording/{st.session_state['user_id']}")

        # Print raw response to debug
        st.code(response.text)
        st.code(response.status_code)


        response_data = response.json()
        st.success("Done!")
        st.markdown(f"**📝 Transcription:** `{response}`")
    except Exception as e:
        st.error(f"Error calling FastAPI: {e}")
