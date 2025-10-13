import { useChatStore } from '@/store/chatStore';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from './ui/card';
import { Slider } from './ui/slider';

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  description?: string;
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
  description,
}: SliderControlProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-gray-700">{label}</label>
        <span className="text-sm font-semibold text-primary-600">{value}</span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(v[0])}
      />
      {description && <p className="text-xs text-gray-500">{description}</p>}
    </div>
  );
}

export default function ParameterControls() {
  const { parameters, updateParameters, selectedModel } = useChatStore();

  // Different providers have different temperature ranges
  const isClaudeModel = selectedModel.startsWith('claude-');
  const temperatureMax = isClaudeModel ? 1 : 2;

  // Clamp temperature if it exceeds the model's max
  const clampedTemperature = Math.min(parameters.temperature, temperatureMax);
  if (clampedTemperature !== parameters.temperature) {
    updateParameters({ temperature: clampedTemperature });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
        <CardDescription>Adjust model generation settings</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <SliderControl
          label="Temperature"
          value={parameters.temperature}
          min={0}
          max={temperatureMax}
          step={0.1}
          onChange={(temperature) => updateParameters({ temperature })}
          description={`Controls randomness. ${isClaudeModel ? 'Claude: 0-1' : 'OpenAI: 0-2'}. Lower = focused, higher = creative.`}
        />

        <SliderControl
          label="Max Tokens"
          value={parameters.maxTokens}
          min={100}
          max={4096}
          step={100}
          onChange={(maxTokens) => updateParameters({ maxTokens })}
          description="Maximum length of the response."
        />

        <SliderControl
          label="Top P"
          value={parameters.topP}
          min={0}
          max={1}
          step={0.05}
          onChange={(topP) => updateParameters({ topP })}
          description="Nucleus sampling. Lower = more deterministic outputs."
        />

        {parameters.frequencyPenalty !== undefined && (
          <SliderControl
            label="Frequency Penalty"
            value={parameters.frequencyPenalty}
            min={-2}
            max={2}
            step={0.1}
            onChange={(frequencyPenalty) => updateParameters({ frequencyPenalty })}
            description="Penalizes repeated tokens based on frequency."
          />
        )}

        {parameters.presencePenalty !== undefined && (
          <SliderControl
            label="Presence Penalty"
            value={parameters.presencePenalty}
            min={-2}
            max={2}
            step={0.1}
            onChange={(presencePenalty) => updateParameters({ presencePenalty })}
            description="Penalizes repeated tokens regardless of frequency."
          />
        )}
      </CardContent>
    </Card>
  );
}
