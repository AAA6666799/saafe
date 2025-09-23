import React from 'react'
import { createRoot } from 'react-dom/client'
import SaafeLovable from './components/SaafeLovable'
import './styles/global.css'

const root = createRoot(document.getElementById('root')!)
root.render(<SaafeLovable/>)
