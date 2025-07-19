import { NextRequest, NextResponse } from 'next/server'

// In-memory storage for demo purposes
// In production, this would be stored in a database
let scanStats = {
  totalScans: 0,
  threatsDetected: 0,
  successRate: 98.1,
  responseTime: 0.8
}

export async function GET() {
  return NextResponse.json(scanStats)
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action, count = 1 } = body

    switch (action) {
      case 'increment_scans':
        scanStats.totalScans += count
        break
      case 'increment_threats':
        scanStats.threatsDetected += count
        break
      case 'update_stats':
        scanStats = { ...scanStats, ...body.stats }
        break
      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        )
    }

    return NextResponse.json(scanStats)
  } catch (error) {
    console.error('Stats update error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
} 