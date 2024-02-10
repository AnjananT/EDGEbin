import React, {useState, useEffect} from 'react';
import axios from 'axios';
import './TrashCounter.css'; 
import socketIOClient from 'socket.io-client';

const ENDPOINT = 'http://localhost:5000';

const TrashCounter = ({wasteType}) => {
    const [count, setCount] = useState(0);
    const MAX_COUNT = 40; 

    const fetchStats = async () => {
        try {
            const response = await axios.get('http://localhost:5000/stats');
            const currentCount = response.data[wasteType] || 0;
            setCount(Math.min(currentCount, MAX_COUNT));
        } catch (error){
            console.error('Error fetching stats', error);
        }
        
    };
    
    const handleEmptyClick = async () => {
        setCount(0);
        try {
            await axios.post('http://localhost:5000/reset_stats', {trash_class: wasteType})
        } catch (error){
            console.error('Error resetting stats', error);
        }
    };

    useEffect(() => {
        const socket = socketIOClient(ENDPOINT);

        socket.on('connect', () => {
            console.log('connected to server');
        });

        socket.on('connect_error', (error) => {
            console.error('Socket connection error', error);
        });

        socket.on('stats_update', (updatedStats)=> {
            const currentCount = updatedStats[wasteType] || 0;
            console.log(`current count: ${currentCount}`)
            setCount(Math.min(currentCount, MAX_COUNT));
        });
        
        fetchStats();
        
        return() => {
            socket.disconnect();
        };
    }, [wasteType]);
    const progressBarHeight = (count/MAX_COUNT) * 100 ;
    console.log(`bar height: ${progressBarHeight}`)
    return (
        <div className="counter">
            <div className="progress-container">
                <div className="progress" style={{height: `${progressBarHeight}%`, transition:'height 0.2s ease-in-out'}}></div>
            </div>
            <div className="waste-type">{wasteType}</div>
            <button className = "empty-button" onClick = {handleEmptyClick}>Empty</button>
        </div>
     );
};
export default TrashCounter;