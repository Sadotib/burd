// file: lib/utils/state/bird_result.dart
class BirdResult {
  static final BirdResult _instance = BirdResult._internal();

  factory BirdResult() => _instance;

  BirdResult._internal();

  late String prediction;
  late double confidence ;
}


