import { Siren } from "lucide-react"

interface FlashingSirenProps {
  isInjection: boolean
  size?: number
}

export function FlashingSiren({ isInjection, size = 48 }: FlashingSirenProps) {
  const animationClass = isInjection 
    ? "siren-flash-red" 
    : "siren-flash-green"
  
  const lightColor = isInjection ? "#ef4444" : "#22c55e"
  const lightGlow = isInjection ? "rgba(239, 68, 68, 0.6)" : "rgba(34, 197, 94, 0.6)"

  const lightStyle = {
    animation: isInjection 
      ? 'emergency-light-red 0.8s ease-in-out infinite' 
      : 'emergency-light-green 1.2s ease-in-out infinite',
    backgroundColor: lightColor,
    boxShadow: `0 0 10px ${lightGlow}`,
  }

  const intenseLightStyle = {
    animation: isInjection 
      ? 'emergency-light-red-intense 0.6s ease-in-out infinite' 
      : 'emergency-light-green-intense 0.8s ease-in-out infinite',
    backgroundColor: lightColor,
    boxShadow: `0 0 15px ${lightGlow}`,
  }

  return (
    <div className="relative flex items-center justify-center">
      {/* Flashing Lights */}
      <div className="absolute inset-0 flex items-center justify-center">
        {/* Top Light */}
        <div 
          className="absolute -top-2 w-3 h-3 rounded-full" 
          style={{ ...lightStyle, animationDelay: '0s' }}
        ></div>
        
        {/* Bottom Light */}
        <div 
          className="absolute -bottom-2 w-3 h-3 rounded-full" 
          style={{ ...lightStyle, animationDelay: '0.4s' }}
        ></div>
        
        {/* Left Light */}
        <div 
          className="absolute -left-2 w-3 h-3 rounded-full" 
          style={{ ...lightStyle, animationDelay: '0.2s' }}
        ></div>
        
        {/* Right Light */}
        <div 
          className="absolute -right-2 w-3 h-3 rounded-full" 
          style={{ ...lightStyle, animationDelay: '0.6s' }}
        ></div>
        
        {/* Diagonal Lights - More Intense */}
        <div 
          className="absolute -top-1 -left-1 w-2 h-2 rounded-full" 
          style={{ ...intenseLightStyle, animationDelay: '0.1s' }}
        ></div>
        
        <div 
          className="absolute -top-1 -right-1 w-2 h-2 rounded-full" 
          style={{ ...intenseLightStyle, animationDelay: '0.3s' }}
        ></div>
        
        <div 
          className="absolute -bottom-1 -left-1 w-2 h-2 rounded-full" 
          style={{ ...intenseLightStyle, animationDelay: '0.5s' }}
        ></div>
        
        <div 
          className="absolute -bottom-1 -right-1 w-2 h-2 rounded-full" 
          style={{ ...intenseLightStyle, animationDelay: '0.7s' }}
        ></div>
        
        {/* Additional Ring of Lights */}
        <div 
          className="absolute -top-3 w-2 h-2 rounded-full" 
          style={{ ...intenseLightStyle, animationDelay: '0.15s' }}
        ></div>
        
        <div 
          className="absolute -bottom-3 w-2 h-2 rounded-full" 
          style={{ ...intenseLightStyle, animationDelay: '0.55s' }}
        ></div>
        
        <div 
          className="absolute -left-3 w-2 h-2 rounded-full" 
          style={{ ...intenseLightStyle, animationDelay: '0.25s' }}
        ></div>
        
        <div 
          className="absolute -right-3 w-2 h-2 rounded-full" 
          style={{ ...intenseLightStyle, animationDelay: '0.65s' }}
        ></div>
      </div>
      
      {/* Main Siren Icon */}
      <Siren 
        className={`${animationClass} drop-shadow-lg relative z-10`} 
        size={size}
      />
    </div>
  )
} 