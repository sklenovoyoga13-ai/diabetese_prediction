import hashlib
import secrets
import streamlit as st
from database import User, get_db_session, init_db


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, password_hash = stored_hash.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except ValueError:
        return False


def create_user(username: str, password: str, email: str = None) -> tuple[bool, str]:
    db = get_db_session()
    if not db:
        return False, "Database not available"
    
    try:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            db.close()
            return False, "Username already exists"
        
        if email:
            existing_email = db.query(User).filter(User.email == email).first()
            if existing_email:
                db.close()
                return False, "Email already registered"
        
        new_user = User(
            username=username,
            password_hash=hash_password(password),
            email=email
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        db.close()
        return True, "Account created successfully"
    except Exception as e:
        db.rollback()
        db.close()
        return False, f"Error creating account: {str(e)}"


def authenticate_user(username: str, password: str) -> tuple[bool, User | None]:
    db = get_db_session()
    if not db:
        return False, None
    
    try:
        user = db.query(User).filter(User.username == username).first()
        if user and verify_password(password, user.password_hash):
            db.close()
            return True, user
        db.close()
        return False, None
    except Exception as e:
        db.close()
        return False, None


def get_user_by_id(user_id: int) -> User | None:
    db = get_db_session()
    if not db:
        return None
    
    try:
        user = db.query(User).filter(User.id == user_id).first()
        db.close()
        return user
    except Exception:
        db.close()
        return None


def login_user(user: User):
    st.session_state['logged_in'] = True
    st.session_state['user_id'] = user.id
    st.session_state['username'] = user.username


def logout_user():
    st.session_state['logged_in'] = False
    st.session_state['user_id'] = None
    st.session_state['username'] = None
    if 'prediction_result' in st.session_state:
        del st.session_state['prediction_result']
    if 'user_data' in st.session_state:
        del st.session_state['user_data']


def is_logged_in() -> bool:
    return st.session_state.get('logged_in', False)


def get_current_user_id() -> int | None:
    return st.session_state.get('user_id', None)


def get_current_username() -> str | None:
    return st.session_state.get('username', None)


def render_auth_ui():
    if is_logged_in():
        st.sidebar.markdown(f"**Welcome, {get_current_username()}!**")
        if st.sidebar.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()
        return True
    
    auth_tab = st.sidebar.radio("Account", ["Login", "Sign Up"], horizontal=True)
    
    if auth_tab == "Login":
        st.sidebar.markdown("### Login")
        login_username = st.sidebar.text_input("Username", key="login_username")
        login_password = st.sidebar.text_input("Password", type="password", key="login_password")
        
        if st.sidebar.button("Login", type="primary", use_container_width=True):
            if login_username and login_password:
                success, user = authenticate_user(login_username, login_password)
                if success and user:
                    login_user(user)
                    st.sidebar.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password")
            else:
                st.sidebar.warning("Please enter username and password")
    
    else:
        st.sidebar.markdown("### Create Account")
        signup_username = st.sidebar.text_input("Username", key="signup_username")
        signup_email = st.sidebar.text_input("Email (optional)", key="signup_email")
        signup_password = st.sidebar.text_input("Password", type="password", key="signup_password")
        signup_confirm = st.sidebar.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.sidebar.button("Sign Up", type="primary", use_container_width=True):
            if not signup_username or not signup_password:
                st.sidebar.warning("Please enter username and password")
            elif len(signup_password) < 6:
                st.sidebar.warning("Password must be at least 6 characters")
            elif signup_password != signup_confirm:
                st.sidebar.error("Passwords don't match")
            else:
                success, message = create_user(signup_username, signup_password, signup_email if signup_email else None)
                if success:
                    st.sidebar.success(message + " Please login.")
                else:
                    st.sidebar.error(message)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Create an account to save your predictions and track your health progress over time.")
    
    return False
