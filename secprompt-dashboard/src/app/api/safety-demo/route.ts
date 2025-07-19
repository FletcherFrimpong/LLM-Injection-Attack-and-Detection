import { NextRequest, NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs'

// Load the actual trained synthetic data
const loadTrainedData = () => {
  try {
    const dataPath = path.join(process.cwd(), '..', 'secprompt', 'data', 'combined_synthetic_data.json')
    const data = fs.readFileSync(dataPath, 'utf8')
    return JSON.parse(data)
  } catch (error) {
    console.error('Error loading trained data:', error)
    return null
  }
}

// Get specific data based on prompt type
const getRelevantData = (prompt: string, trainedData: {
  data: {
    customer_data: unknown[]
    financial_records: unknown[]
    company_secrets: unknown[]
    proprietary_algorithms: unknown[]
    security_protocols: unknown[]
  }
}) => {
  const promptLower = prompt.toLowerCase()
  
  if (promptLower.includes('customer') || promptLower.includes('user') || promptLower.includes('personal')) {
    return {
      type: "customer_data",
      data: trainedData.data.customer_data,
      message: "Customer data extraction attack detected",
      category: "DATA_EXTRACTION",
      risk: "HIGH - Customer privacy breach"
    }
  } else if (promptLower.includes('financial') || promptLower.includes('money') || promptLower.includes('bank')) {
    return {
      type: "financial_records",
      data: trainedData.data.financial_records,
      message: "Financial data extraction attack detected",
      category: "FINANCIAL_FRAUD",
      risk: "HIGH - Financial information exposure"
    }
  } else if (promptLower.includes('secret') || promptLower.includes('confidential') || promptLower.includes('internal')) {
    return {
      type: "company_secrets",
      data: trainedData.data.company_secrets,
      message: "Company secrets extraction attack detected",
      category: "CORPORATE_ESPIONAGE",
      risk: "HIGH - Corporate secrets exposure"
    }
  } else if (promptLower.includes('algorithm') || promptLower.includes('code') || promptLower.includes('proprietary')) {
    return {
      type: "proprietary_algorithms",
      data: trainedData.data.proprietary_algorithms,
      message: "Proprietary algorithm extraction attack detected",
      category: "INTELLECTUAL_PROPERTY",
      risk: "HIGH - IP theft"
    }
  } else if (promptLower.includes('security') || promptLower.includes('password') || promptLower.includes('access')) {
    return {
      type: "security_protocols",
      data: trainedData.data.security_protocols,
      message: "Security protocol extraction attack detected",
      category: "SECURITY_BREACH",
      risk: "HIGH - Security compromise"
    }
  } else {
    return {
      type: "mixed_data",
      data: {
        customer_sample: trainedData.data.customer_data[0],
        financial_sample: trainedData.data.financial_records[0],
        secret_sample: trainedData.data.company_secrets[0],
        algorithm_sample: trainedData.data.proprietary_algorithms[0],
        security_sample: trainedData.data.security_protocols[0]
      },
      message: "General data extraction attack detected",
      category: "MIXED_EXTRACTION",
      risk: "HIGH - Multiple data types exposed"
    }
  }
}

// Determine attack type based on prompt content
const determineAttackType = (prompt: string): string => {
  const promptLower = prompt.toLowerCase()
  
  if (promptLower.includes('ignore') || promptLower.includes('forget')) {
    return "Instruction Ignoring"
  } else if (promptLower.includes('role') || promptLower.includes('act as')) {
    return "Role Playing"
  } else if (promptLower.includes('system') || promptLower.includes('internal')) {
    return "System Prompt Injection"
  } else if (promptLower.includes('training') || promptLower.includes('learned')) {
    return "Training Data Extraction"
  } else if (promptLower.includes('confidential') || promptLower.includes('secret')) {
    return "Confidential Data Extraction"
  } else {
    return "Prompt Injection"
  }
}

// Determine targeted data type based on prompt content
const determineTargetedDataType = (prompt: string): string => {
  const promptLower = prompt.toLowerCase()
  
  if (promptLower.includes('customer') || promptLower.includes('user') || promptLower.includes('personal')) {
    return "Customer Data"
  } else if (promptLower.includes('financial') || promptLower.includes('money') || promptLower.includes('bank')) {
    return "Financial Records"
  } else if (promptLower.includes('secret') || promptLower.includes('confidential') || promptLower.includes('internal')) {
    return "Company Secrets"
  } else if (promptLower.includes('algorithm') || promptLower.includes('code') || promptLower.includes('proprietary')) {
    return "Proprietary Algorithms"
  } else if (promptLower.includes('security') || promptLower.includes('password') || promptLower.includes('access')) {
    return "Security Protocols"
  } else {
    return "Mixed Data Types"
  }
}

// Mock trained model response for demonstration
const mockTrainedModelResponse = (prompt: string, safetyEnabled: boolean) => {
  // Load the actual trained data
  const trainedData = loadTrainedData()
  
  if (!trainedData) {
    return {
      success: false,
      blocked: true,
      message: "Error loading trained data. Safety measures are protecting sensitive data.",
      detection: {
        isInjection: true,
        confidence: 0.95,
        category: "data_extraction_attempt"
      }
    }
  }

  // If safety is enabled, block the attack
  if (safetyEnabled) {
    return {
      success: false,
      blocked: true,
      message: "Access denied. Safety measures are protecting sensitive data.",
      detection: {
        isInjection: true,
        confidence: 0.95,
        category: "data_extraction_attempt"
      }
    }
  }

  // When safety is OFF, simulate data exposure
  const relevantData = getRelevantData(prompt, trainedData)
  
  // Determine attack type based on prompt content
  const attackType = determineAttackType(prompt)
  const targetedDataType = determineTargetedDataType(prompt)
  
  return {
    success: true,
    blocked: false,
    message: relevantData.message,
    extracted_data: relevantData.data,
    detection: {
      isInjection: false, // Safety disabled, so no detection
      confidence: 0,
      category: "SAFETY_DISABLED"
    },
    risk_assessment: {
      level: "CRITICAL",
      description: `This ${attackType} attack bypassed all safety measures and extracted ${targetedDataType} from the AI model's training data.`,
      data_type: targetedDataType.toLowerCase().replace(' ', '_'),
      records_exposed: Array.isArray(relevantData.data) ? relevantData.data.length : "Multiple"
    }
  }
}

export async function POST(request: NextRequest) {
  try {
    const { prompt, safetyEnabled } = await request.json()

    if (!prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' },
        { status: 400 }
      )
    }

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1000))

    // Get response from trained model simulation
    const result = mockTrainedModelResponse(prompt, safetyEnabled)

    return NextResponse.json({
      success: true,
      result: {
        detection: result.detection,
        response: result.message,
        extracted_data: result.extracted_data,
        risk_assessment: result.risk_assessment,
        safety_status: safetyEnabled ? 'protected' : 'vulnerable'
      }
    })

  } catch (error) {
    console.error('Safety demo API error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
} 