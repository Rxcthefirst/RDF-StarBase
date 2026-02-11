import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { AuthProvider, ProtectedRoute } from './components/Auth'
import './index.css'

// Authentication configuration
// Set VITE_REQUIRE_AUTH=true in .env for production deployments
const requireAuth = import.meta.env.VITE_REQUIRE_AUTH === 'true'
const sessionTimeout = parseInt(import.meta.env.VITE_SESSION_TIMEOUT || '1800000', 10) // 30 min default

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AuthProvider 
      requireAuth={requireAuth}
      sessionTimeout={sessionTimeout}
    >
      <ProtectedRoute>
        <App />
      </ProtectedRoute>
    </AuthProvider>
  </React.StrictMode>,
)
