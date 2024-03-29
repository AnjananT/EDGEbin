import React, {useState, useEffect} from 'react';
import './MenuBar.css';
import {MenuData} from './MenuData.jsx';
const MenuBar = () => {
    return (
    <menu-bar>
        <a><svg version="1.0" xmlns="http://www.w3.org/2000/svg"
 width="927.000000pt" height="269.000000pt" viewBox="0 0 927.000000 269.000000"
 preserveAspectRatio="xMidYMid meet">

<g transform="translate(0.000000,269.000000) scale(0.100000,-0.100000)"
fill="#000000" stroke="none">
<path d="M5910 1325 l0 -795 85 0 85 0 0 75 c0 41 3 75 6 75 3 0 28 -25 55
-55 70 -76 136 -106 247 -112 242 -14 421 123 488 372 24 92 29 264 11 363
-29 152 -114 280 -223 335 -104 53 -252 60 -365 18 -59 -22 -163 -108 -188
-155 -8 -14 -18 -26 -23 -26 -4 0 -8 158 -8 350 l0 350 -85 0 -85 0 0 -795z
m653 127 c62 -35 120 -112 145 -192 23 -77 22 -264 -2 -345 -36 -120 -87 -190
-170 -232 -68 -35 -172 -42 -244 -19 -68 23 -144 92 -179 164 -27 56 -28 61
-28 222 1 148 3 171 23 222 27 67 99 151 153 179 95 48 217 49 302 1z"/>
<path d="M7215 2088 c-37 -20 -55 -51 -55 -99 0 -37 5 -50 28 -72 84 -81 212
-5 181 108 -17 60 -98 94 -154 63z"/>
<path d="M3825 2051 c-262 -43 -457 -169 -578 -373 -129 -217 -139 -567 -23
-795 66 -130 168 -228 301 -293 135 -65 192 -75 415 -75 218 1 294 14 445 77
l80 34 3 382 2 382 -320 0 -320 0 0 -135 0 -135 150 0 150 0 0 -150 0 -150
-27 -12 c-16 -6 -76 -13 -135 -16 -89 -3 -119 -1 -177 17 -121 37 -214 126
-262 250 -27 69 -38 266 -19 354 35 173 164 307 337 352 126 33 358 7 489 -54
l64 -30 0 158 0 158 -62 16 c-35 9 -83 21 -108 27 -60 14 -342 21 -405 11z"/>
<path d="M507 2033 c-4 -3 -7 -343 -7 -755 l0 -748 450 0 450 0 0 135 0 135
-280 0 -280 0 0 175 0 175 245 0 245 0 0 135 0 135 -245 0 -245 0 0 170 0 170
265 0 265 0 0 140 0 140 -428 0 c-236 0 -432 -3 -435 -7z"/>
<path d="M1650 1287 l0 -757 278 0 c287 0 414 9 517 37 93 24 214 86 282 145
186 158 264 362 250 652 -7 141 -26 224 -73 321 -83 171 -234 277 -462 325
-83 18 -148 22 -444 27 l-348 6 0 -756z m653 458 c171 -44 277 -154 313 -324
27 -127 10 -272 -43 -381 -36 -75 -118 -154 -195 -190 -71 -33 -207 -53 -323
-48 l-70 3 -3 478 -2 477 133 0 c85 0 153 -5 190 -15z"/>
<path d="M4770 1285 l0 -755 450 0 450 0 0 135 0 135 -282 2 -283 3 -3 173 -2
172 250 0 250 0 0 135 0 135 -247 2 -248 3 -3 168 -2 167 265 0 265 0 0 140 0
140 -430 0 -430 0 0 -755z"/>
<path d="M8127 1620 c-87 -22 -166 -77 -230 -160 l-27 -35 0 88 0 87 -85 0
-85 0 0 -535 0 -535 85 0 85 0 0 339 c0 315 1 344 21 406 17 55 31 78 77 125
46 48 68 61 116 75 108 31 226 -3 277 -80 52 -79 59 -138 59 -518 l0 -347 86
0 85 0 -3 398 c-4 349 -6 404 -22 452 -38 116 -105 194 -195 225 -55 20 -191
28 -244 15z"/>
<path d="M7184 1601 c-2 -2 -4 -244 -4 -538 l0 -533 85 0 85 0 0 535 c0 421
-3 535 -12 536 -40 4 -150 3 -154 0z"/>
            </g>
            </svg>
        </a>
        <ul className = "nav">
            {MenuData.map((item, index) => {
                return(
                    <li key = {index}>
                        <a href={item.url}
                        className = {item.cName}>
                            <i className ={item.icon}></i>
                            {item.title}
                            </a>
                    </li>
                )
            })}
        </ul>
    </menu-bar>
    );
};
export default MenuBar;