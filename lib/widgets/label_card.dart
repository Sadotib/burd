import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:turd/utils/helper/helper.dart';


class LabelCard extends StatefulWidget {
  final String label;
  final String imagePath;

  const LabelCard({super.key, required this.label, required this.imagePath});

  @override
  State<LabelCard> createState() => _LabelCardState();
}

class _LabelCardState extends State<LabelCard> with SingleTickerProviderStateMixin {
  bool isExpanded = false;

  @override
  Widget build(BuildContext context) {
    return Card(
      shadowColor: Color(0xFF1F4BEA),
      elevation: 3,
      surfaceTintColor: Color(0xFF1F4BEA),
      color: HelperFunction.isDark(context)?Colors.black:Colors.white,
      margin: const EdgeInsets.all(12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: () {
          setState(() => isExpanded = !isExpanded);
        },
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                widget.label,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.bold,
                  color: HelperFunction.isDark(context)?Colors.white:Colors.black,
                ),
              ),
              AnimatedCrossFade(
                firstChild: const SizedBox.shrink(),
                secondChild: Column(
                  children: [
                    const SizedBox(height: 12),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: CachedNetworkImage(
                        
                        imageUrl: widget.imagePath,
                        placeholder: (context,url) => const CircularProgressIndicator(),
                        
                        width: double.infinity,
                        height: 200,
                        fit: BoxFit.contain,
                        errorWidget: (context, url, error) => Icon(Icons.broken_image),
                        // errorBuilder: (_, __, ___) => const Icon(Icons.broken_image),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'This is a sample description for ${widget.label}.',
                      style: TextStyle(color: Theme.of(context).textTheme.bodyMedium?.color),
                    ),
                  ],
                ),
                crossFadeState: isExpanded ? CrossFadeState.showSecond : CrossFadeState.showFirst,
                duration: const Duration(milliseconds: 300),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

