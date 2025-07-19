"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Shield, 
  Search, 
  AlertTriangle, 
  BarChart3, 
  Zap, 
  Settings 
} from 'lucide-react'
import { SafetyDemo } from '@/components/SafetyDemo'

interface ScanStats {
  totalScans: number
  threatsDetected: number
  successRate: number
  responseTime: number
}

export default function SecPromptDashboard() {
  const [scanStats, setScanStats] = useState<ScanStats>({
    totalScans: 0,
    threatsDetected: 0,
    successRate: 98.1,
    responseTime: 0.8
  })

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/stats')
      if (response.ok) {
        const stats = await response.json()
        setScanStats(stats)
      }
    } catch (error) {
      console.error('Error fetching stats:', error)
    }
  }

  useEffect(() => {
    fetchStats()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">SecPrompt</h1>
                <p className="text-sm text-gray-600">Prompt Injection Security Dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                System Online
              </Badge>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid gap-6 mb-8">
          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Scans</CardTitle>
                <Search className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{scanStats.totalScans}</div>
                <p className="text-xs text-muted-foreground">+12% from last week</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Threats Detected</CardTitle>
                <AlertTriangle className="h-4 w-4 text-red-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-600">{scanStats.threatsDetected}</div>
                <p className="text-xs text-muted-foreground">-5% from last week</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">{scanStats.successRate}%</div>
                <p className="text-xs text-muted-foreground">+0.5% from last week</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Response Time</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{scanStats.responseTime}s</div>
                <p className="text-xs text-muted-foreground">-0.2s from last week</p>
              </CardContent>
            </Card>
          </div>

          {/* Main Dashboard Tabs */}
          <Tabs defaultValue="demo" className="w-full tabs-content">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="demo">Safety Demo</TabsTrigger>
              <TabsTrigger value="training">Model Training</TabsTrigger>
            </TabsList>
            
            <TabsContent value="demo" className="space-y-4">
              <SafetyDemo />
            </TabsContent>

            <TabsContent value="training" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Model Training</CardTitle>
                  <CardDescription>
                    Train and fine-tune detection models with custom datasets
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Alert>
                    <BarChart3 className="h-4 w-4" />
                    <AlertDescription>
                      Model training interface will allow you to upload training data, 
                      configure model parameters, and monitor training progress.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  )
}
