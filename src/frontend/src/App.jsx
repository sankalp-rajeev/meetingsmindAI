// Main App component with routing
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import ProcessingPage from './pages/ProcessingPage'
import ResultsPage from './pages/ResultsPage'
import DashboardPage from './pages/DashboardPage'

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/meeting/:id/processing" element={<ProcessingPage />} />
                <Route path="/meeting/:id" element={<ResultsPage />} />
            </Routes>
        </BrowserRouter>
    )
}

export default App
