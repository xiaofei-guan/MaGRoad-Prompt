/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // Include all JS, TS, JSX, TSX files in src
  ],
  theme: {
    extend: {
      colors: {
        'custom-red': '#FF5733',
        'custom-blue': {
          light: '#85d7ff', 
          DEFAULT: '#1fb6ff',
          dark: '#009eeb',
        },
      }
    },
  },
  plugins: [],
}