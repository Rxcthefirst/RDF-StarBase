/**
 * Authentication Components
 * 
 * Enterprise SSO authentication for RDF-StarBase UI
 * Supports OIDC/OAuth2 providers (Keycloak, Azure AD, Okta, Auth0)
 * 
 * @example
 * // In App.jsx, wrap your app with AuthProvider
 * import { AuthProvider, ProtectedRoute, UserMenu } from './components/Auth';
 * 
 * function App() {
 *   return (
 *     <AuthProvider requireAuth={process.env.NODE_ENV === 'production'}>
 *       <ProtectedRoute>
 *         <MainContent />
 *       </ProtectedRoute>
 *       <UserMenu />
 *     </AuthProvider>
 *   );
 * }
 */

export { 
  AuthProvider, 
  useAuth, 
  ProtectedRoute, 
  LoginPage, 
  SessionTimeoutWarning,
  UserMenu 
} from './AuthContext';

import './Auth.css';
