import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import TrashCounter from './components/TrashCounter'
import MenuBar from './components/MenuBar'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
    <div>
        <MenuBar></MenuBar>
      </div>
      <div>
        <TrashCounter wasteType="recycle"/>
        <TrashCounter wasteType="trash"/>
        <TrashCounter wasteType="electronic"/>
        <TrashCounter wasteType="organic"/>
      </div>
    </>
  )
}

export default App;
