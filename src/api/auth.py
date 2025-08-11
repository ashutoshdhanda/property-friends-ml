"""API authentication middleware and utilities."""

import secrets
from typing import Optional
from datetime import datetime, timedelta

from fastapi import HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class APIKeyManager:
    """Manages API key validation and JWT tokens."""
    
    def __init__(self):
        self.settings = get_settings()
        self.valid_api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> set:
        """Load valid API keys from configuration."""
        # For production, these should be stored securely (database, vault, etc.)
        # For now, we'll use a configurable secret that can generate keys
        
        # Generate a default API key from the secret
        default_key = self._generate_api_key_from_secret(self.settings.api_key_secret)
        
        return {default_key, "demo-api-key-for-testing"}  # Include a demo key for testing
    
    def _generate_api_key_from_secret(self, secret: str) -> str:
        """Generate a deterministic API key from the secret."""
        import hashlib
        return f"pk_{hashlib.sha256(secret.encode()).hexdigest()[:32]}"
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key."""
        is_valid = api_key in self.valid_api_keys
        
        if is_valid:
            logger.info("api_key_validated_successfully", key_prefix=api_key[:8] + "...")
        else:
            logger.warning("invalid_api_key_attempt", key_prefix=api_key[:8] + "..." if len(api_key) > 8 else "invalid")
        
        return is_valid
    
    def create_access_token(self, data: dict) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.settings.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.settings.secret_key, 
            algorithm=self.settings.algorithm
        )
        
        logger.info("access_token_created", expires_at=expire.isoformat())
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token, 
                self.settings.secret_key, 
                algorithms=[self.settings.algorithm]
            )
            return payload
        except JWTError as e:
            logger.warning("token_verification_failed", error=str(e))
            return None


# Global API key manager instance
api_key_manager = APIKeyManager()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify API key from Authorization header.
    
    Expects: Authorization: Bearer <api-key>
    """
    if not credentials:
        logger.warning("missing_authorization_header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = credentials.credentials
    
    if not api_key_manager.validate_api_key(api_key):
        logger.warning("invalid_api_key_provided", key_prefix=api_key[:8] + "...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key


async def get_current_user(api_key: str = Depends(verify_api_key)) -> dict:
    """Get current user information based on API key."""
    # For now, we'll return basic user info
    # In production, this would lookup user details from database
    return {
        "api_key_prefix": api_key[:8] + "...",
        "authenticated": True,
        "permissions": ["predict", "batch_predict"]
    }


def generate_new_api_key() -> str:
    """Generate a new secure API key."""
    # Generate a secure random API key
    return f"pk_{secrets.token_hex(32)}"


def get_demo_credentials() -> dict:
    """Get demo credentials for testing."""
    return {
        "api_key": "demo-api-key-for-testing",
        "description": "Demo API key for testing the endpoints"
    }
